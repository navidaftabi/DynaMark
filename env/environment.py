# env/environment.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .plants.base import PlantBase
from .core.detector import ChiSquareDetector
from .core.belief import ReplayBeliefFilter
from .core.beta_models import (
    BetaLookup, 
    BetaLookupConfig, 
    ChiSquareBetaMC, 
    MCBetaConfig
    )
from .core.covariance import cov_from_action


def _as_col(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a
    if a.ndim == 2 and a.shape[0] == 1:
        return a.reshape(-1, 1)
    raise ValueError(f"expected scalar/1d/col, got {a.shape}")


def _chol_action_dim(d: int) -> int:
    return int(d * (d + 1) // 2)


class DynaMarkEnv(gym.Env):
    """
    Unified env. If plant.step() returns ctx['batch'], we treat one env.step()
    as one decision epoch and internally iterate detector/belief updates.
      - batch uses FIRST proc_len samples
      - sample 0 is anchor (no test/update); loop starts at i=1
    """
    metadata = {"render_modes": []}

    def __init__(self, plant: PlantBase, cfg: Dict[str, Any]):
        super().__init__()
        self.plant = plant
        self.cfg = cfg

        self.d = int(getattr(plant, "u_dim", 1))
        self.n = int(getattr(plant, "y_dim", 1))

        reward_cfg = cfg.get("reward", cfg)
        self.w1 = float(reward_cfg.get("w1", 0.35))
        self.w2 = float(reward_cfg.get("w2", 0.25))
        self.w3 = float(reward_cfg.get("w3", 0.40))

        det_cfg = cfg.get("detector", cfg)
        self.alpha = float(det_cfg.get("alpha", cfg.get("alpha", 0.01)))
        g_tilde_override = det_cfg.get("g_tilde_override", None)

        bel_cfg = cfg.get("belief", cfg)
        q_prior = float(bel_cfg.get("q_prior", bel_cfg.get("q", cfg.get("q", 0.05))))
        p_geom = float(bel_cfg.get("p_geom", cfg.get("p_geom", 1e-3)))

        beta_cfg_in = cfg.get("beta", {})
        beta_mode = str(beta_cfg_in.get("mode", "mc_gaussian")).lower()

        if beta_mode in {"mc_gaussian", "mc", "gaussian"}:
            beta_kwargs = dict(beta_cfg_in)
            beta_kwargs.pop("mode", None)
            beta_kwargs.setdefault("alpha", self.alpha)
            beta_kwargs.setdefault("p_geom", p_geom)
            beta_kwargs.setdefault("delta_t", 1)
            beta_kwargs.setdefault("n_mc", 1000)
            beta_kwargs.setdefault("seed", int(cfg.get("beta_seed", 0)))

            beta_cfg = MCBetaConfig(**beta_kwargs)
            self.beta_model = ChiSquareBetaMC(dof=self.n, u_dim=self.d, cfg=beta_cfg)
            if g_tilde_override is not None:
                detector_g_tilde = float(g_tilde_override)
            elif self.alpha is not None:
                detector_g_tilde = None
            else:
                print("Warning: no alpha or g_tilde provided for detector. Using default alpha=0.01 for detector threshold.")
                detector_g_tilde = None
                self.alpha = 0.01

        elif beta_mode in {"lookup", "nongaussian_lookup"}:
            beta_cfg = BetaLookupConfig(
                lookup_path=beta_cfg_in["lookup_path"],
                clip=beta_cfg_in.get("clip", True),
                eps=beta_cfg_in.get("eps", 1e-12),
            )
            self.beta_model = BetaLookup(dof=self.n, u_dim=self.d, cfg=beta_cfg)
            if g_tilde_override is not None:
                detector_g_tilde = float(g_tilde_override)
            elif self.beta_model.g_tilde is not None:
                detector_g_tilde = self.beta_model.g_tilde
            else:
                print("Warning: no g_tilde found in beta lookup; using alpha for detector threshold.")
                detector_g_tilde = None
        else:
            raise ValueError(
                f"Unsupported beta.mode='{beta_mode}'. "
                "Use 'mc_gaussian' or 'lookup'."
            )
        
        self.detector = ChiSquareDetector(
            alpha=None if detector_g_tilde is not None else self.alpha,
            dof=self.n,
            g_tilde=detector_g_tilde
        )

        self.belief = ReplayBeliefFilter(
            q_prior=q_prior,
            alpha=self.alpha,
            p_geom=p_geom
        )

        act_dim = _chol_action_dim(self.d)
        low = np.full((act_dim,), -50.0, dtype=np.float32)
        high = np.full((act_dim,), 50.0, dtype=np.float32)
        low[: self.d] = 0.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n + 1,), dtype=np.float32
        )

        self.t_dec = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.t_dec = 0
        self.plant.reset(seed=seed)

        if hasattr(self.detector, "reset"):
            self.detector.reset()
        if hasattr(self.belief, "reset"):
            self.belief.reset()
        if hasattr(self.beta_model, "reset"):
            self.beta_model.reset()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        y = np.asarray(self.plant.y_curr, dtype=float).reshape(-1)
        S1 = float(getattr(self.belief, "S1", 0.0))
        return np.concatenate([y, np.array([S1], dtype=float)]).astype(np.float32)

    def _reward_terms(self, phi: np.ndarray, xdiff: np.ndarray, S1: float) -> Tuple[float, float, float, float]:
        phi_cost = float(np.sum(np.abs(phi)))
        delta_cost = float(np.linalg.norm(xdiff))
        eps = 1e-12
        p = float(np.clip(S1, eps, 1.0 - eps))
        detection = np.abs(0.5 - p)
        reward = - self.w1 * phi_cost - self.w2 * delta_cost + self.w3 * detection
        return reward, phi_cost, delta_cost, detection

    def step(self, action, *, next_y_override=None):
        self.t_dec += 1

        U = cov_from_action(np.asarray(action, dtype=float), self.d)
        out = self.plant.step(U, next_y_override=next_y_override)
        ctx = getattr(out, "ctx", {}) or {}

        #  batched (SM-DT discrete)
        if "batch" in ctx and ctx["batch"]:
            batch = ctx["batch"]
            L = int(batch.get("proc_len", 0))
            if L <= 0:
                obs = self._get_obs()
                return obs, 0.0, True, False, {"batched": True, "reason": "empty_batch"}

            x = np.asarray(batch["x_curr"], dtype=float).reshape(L)
            xw = np.asarray(batch["xw_curr"], dtype=float).reshape(L)
            phi = np.asarray(batch["phi"], dtype=float).reshape(L)
            r = np.asarray(batch["r"], dtype=float).reshape(L)
            Q = np.asarray(batch["Q"], dtype=float).reshape(L)
            B = np.asarray(batch["B"], dtype=float).reshape(L)
            attacked = np.asarray(batch.get("attacked", np.zeros(L)), dtype=int).reshape(L)

            It_seq = np.zeros(L, dtype=int)
            g_seq = np.zeros(L, dtype=float)
            beta_seq = np.full(L, np.nan, dtype=float)
            S1_seq = np.full(L, np.nan, dtype=float)
            rew_seq = np.zeros(L, dtype=float)

            # i=0 (no detector/belief update)
            S1_prior = float(getattr(self.belief, "S1", 0.0))
            rew_seq[0], _, _, _ = self._reward_terms(_as_col(phi[0]), _as_col(xw[0] - x[0]), S1_prior)
            S1_seq[0] = S1_prior

            for i in range(1, L):
                # detector test uses r_i, Q_i
                Q_i = float(Q[i])
                r_i = float(r[i])
                It_i, g_i = self.detector.test(_as_col(r_i), np.array([[Q_i]], dtype=float))
                It_seq[i] = int(It_i)
                g_seq[i] = float(g_i)

                # beta uses history 
                step_ctx_i = {
                    "B": np.array([[float(B[i - 1])]], dtype=float),
                    "H": np.array([[float(B[i - 1])]], dtype=float),
                    "Q": np.array([[float(Q[i - 1])]], dtype=float),
                }
                self.beta_model.push(U=U, step_ctx=step_ctx_i)
                beta_i = float(self.beta_model.beta_t(t_det=int(getattr(self.belief, "t", 0))))
                beta_seq[i] = beta_i

                self.belief.update(It_i, beta_i)
                S1_i = float(getattr(self.belief, "S1", 0.0))
                S1_seq[i] = S1_i

                rew_seq[i], _, _, _ = self._reward_terms(_as_col(phi[i]), _as_col(xw[i] - x[i]), S1_i)

            reward = float(np.sum(rew_seq))
            obs = self._get_obs()
            terminated = bool(getattr(out, "terminated", False))
            truncated = False

            info = {
                "batched": True,
                "proc_len": L,
                "reward_seq": rew_seq,
                "It_seq": It_seq,
                "g_seq": g_seq,
                "beta_seq": beta_seq,
                "S1_seq": S1_seq,
                "attacked_seq": attacked,
                "epoch_attacked": bool(np.any(attacked)),
                "t_fast_start": ctx.get("t_fast_start", None),
                "n_fast": ctx.get("n_fast", None),
            }
            return obs, reward, terminated, truncated, info

        # normal (single-step)
        It, g = self.detector.test(out.r, out.Q)
        self.beta_model.push(U=U, step_ctx=ctx)
        beta_t = self.beta_model.beta_t(t_det=int(getattr(self.belief, "t", 0)))
        self.belief.update(It, beta_t)
        S1 = float(getattr(self.belief, "S1", 0.0))

        reward, phi_cost, delta_cost, info_bonus = self._reward_terms(out.phi, out.x_wowm_curr - out.x_curr, S1)

        obs = self._get_obs()
        terminated = bool(getattr(out, "terminated", False))
        truncated = False
        info = {
            "batched": False,
            "It": int(It),
            "g": float(g),
            "beta": float(beta_t),
            "S1": float(S1),
            "reward": float(reward),
            "phi_cost": float(phi_cost),
            "delta_cost": float(delta_cost),
            "info_bonus": float(info_bonus),
        }
        return obs, float(reward), terminated, truncated, info
