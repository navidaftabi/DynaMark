
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .sm_dt_continuous import SMDTContinuousPlant
from ..core.types import StepOut


class SMDTDiscretePlant(SMDTContinuousPlant):
    """
    Discretized (dual-time / batched) Stepper Motor Digital Twin Plant.

      - Each RL "detector step" (one call to `step`) simulates `plant_block_len`
        fast-time samples using the SM-DT dynamics.

      - The detector/belief update is run on the FIRST `proc_len` samples of
        that block (the "processed" sub-block). The environment (DynaMarkEnv)
        consumes these samples via `ctx["batch"]`.

      - Observation to the agent is based on the last processed measurement,
        NOT the last fast sample.

    Replay / override semantics (compatibility):
      - If `next_y_override` is provided, it can be:
          * scalar: use the same override for every fast step
          * 1D sequence/array: per-fast-step overrides (len >= plant_block_len)
      - For any fast step with an override, we mark that sample as "attacked"
        and apply an actuator-side perturbation consistent with the legacy code:
            u_attack = -2 * u_cmd  -> applied input becomes -u_cmd
        while reporting the observed y as the provided override.
    """

    y_dim = 1
    u_dim = 1

    def __init__(
        self,
        data: Dict[str, Any],
        *,
        seed: Optional[int] = None,
        plant_block_len: int = 500,
        proc_len: int = 100,
        T_fast: Optional[int] = None,
    ):
        super().__init__(data, seed=seed)

        self.plant_block_len = int(plant_block_len)
        self.proc_len = int(proc_len)
        if self.plant_block_len <= 0:
            raise ValueError("plant_block_len must be > 0")
        if self.proc_len <= 0:
            raise ValueError("proc_len must be > 0")

        self.T_fast = None if T_fast is None else int(T_fast)
        if self.T_fast is not None and self.T_fast <= 0:
            raise ValueError("T_fast must be > 0 if provided")

        # fast-time counter (used for truncation/horizon)
        self.fast_total: int = 0

        # what the agent "sees" (last processed y)
        self._y_agent: float = float(getattr(self, "mu0", 0.0))

    @property
    def y_curr(self) -> np.ndarray:
        # current observation to agent as (1,1)
        return np.array([[self._y_agent]], dtype=float)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self.fast_total = 0
        self._y_agent = float(self._y_obs)

    def _get_override_for_step(
        self, next_y_override: Optional[Union[float, Sequence[float], np.ndarray]], b: int
    ) -> Optional[float]:
        if next_y_override is None:
            return None

        # scalar -> broadcast
        if np.isscalar(next_y_override):
            return float(next_y_override)

        arr = np.asarray(next_y_override, dtype=float).reshape(-1)
        if b < arr.size:
            return float(arr[b])
        return None

    def step(self, U: np.ndarray, *, next_y_override=None) -> StepOut:
        """
        One detector step = simulate one fast block of length `plant_block_len`,
        then expose a processed sub-block of length up to `proc_len` via ctx["batch"].
        """
        U = U * (3.3 / 1000)**2
        # if already done, still return a valid StepOut
        if getattr(self, "_terminate", False):
            empty = np.array([], dtype=float)
            ctx = {
                "batch": {
                    "proc_len": 0,
                    "x_curr": empty,
                    "xw_curr": empty,
                    "phi": empty,
                    "r": empty,
                    "Q": empty,
                    "B": empty,
                    "attacked": np.array([], dtype=int),
                },
                "t_fast_start": self.fast_total,
                "n_fast": 0,
                "truncated": bool(self.T_fast is not None and self.fast_total >= self.T_fast),
            }
            z = np.array([[0.0]], dtype=float)
            return StepOut(
                y_prev=z,
                y_curr=z,
                y_wowm_curr=z,
                x_prev=z,
                x_curr=z,
                x_wowm_curr=z,
                u=z,
                phi=z,
                y_hat=z,
                Q=z,
                r=z,
                ctx=ctx,
                terminated=True,
            )

        t_fast_start = int(self.fast_total)

        Y_blk: list[float] = []
        X_blk: list[float] = []
        Xw_blk: list[float] = []
        PHI_blk: list[float] = []
        Ucmd_blk: list[float] = []
        Yhat_blk: list[float] = []
        R_blk: list[float] = []
        Q_blk: list[float] = []
        B_blk: list[float] = []
        attacked_blk: list[int] = []

        # run fast block
        for b in range(self.plant_block_len):
            # truncate by horizon if requested
            if self.T_fast is not None and self.fast_total >= self.T_fast:
                break
            if getattr(self, "_terminate", False):
                break

            # snapshot
            y_prev_obs = float(self._y_obs)
            x_prev_true = float(self._x_true)
            x_prev_wowm = float(self._x_wowm)

            # noises and inputs
            w_t = float(self._sample_w())
            u_wowm = float(self.ut_gmm.sample())
            phi_t = float(self._sample_phi(U))
            u_cmd = u_wowm + phi_t

            # replay override for this fast step (if any)
            y_override = self._get_override_for_step(next_y_override, b)
            attacked = y_override is not None

            # wowm evolution (reference)
            x_wowm_next = (self.A * x_prev_wowm) + (self.B * u_wowm) + w_t

            # true evolution (possibly actuator-side perturbation under replay)
            if attacked:
                # legacy: u_attack = -2*u_cmd -> applied input becomes -u_cmd
                u_applied = u_cmd - 2.0 * u_cmd
                x_true_next = (self.A * x_prev_true) + (self.B * u_applied) + w_t
                y_obs_next = float(y_override)
            else:
                x_true_next = (self.A * x_prev_true) + (self.B * u_cmd) + w_t
                y_obs_next = float(x_true_next)

            # predictor (defender uses commanded u_cmd)
            y_hat = (self.A * y_prev_obs) + (self.B * u_cmd)
            r = y_obs_next - y_hat

            # update internal streams (fast-time)
            self._t += 1
            self.fast_total += 1
            self._x_true = float(x_true_next)
            self._x_wowm = float(x_wowm_next)
            self._y_obs = float(y_obs_next)

            # log current step (parameters correspond to this step)
            Y_blk.append(float(y_obs_next))
            X_blk.append(float(x_true_next))
            Xw_blk.append(float(x_wowm_next))
            PHI_blk.append(float(phi_t))
            Ucmd_blk.append(float(u_cmd))
            Yhat_blk.append(float(y_hat))
            R_blk.append(float(r))
            Q_blk.append(float(self.Q))
            B_blk.append(float(self.B))
            attacked_blk.append(1 if attacked else 0)

            # advance block scheduling (may update A,B,Q,y_bar,ut_gmm for next step)
            self._advance_block()

        n_fast = len(Y_blk)

        # processed sub-block (first proc_len)
        L = int(min(self.proc_len, n_fast))

        if L > 0:
            self._y_agent = float(Y_blk[L - 1])
        else:
            # nothing processed; fall back to current observed y
            self._y_agent = float(self._y_obs)

        batch = {
            "proc_len": L,
            "y": np.asarray(Y_blk[:L], dtype=float),
            "y_hat": np.asarray(Yhat_blk[:L], dtype=float),
            "u": np.asarray(Ucmd_blk[:L], dtype=float),
            "x_curr": np.asarray(X_blk[:L], dtype=float),
            "xw_curr": np.asarray(Xw_blk[:L], dtype=float),
            "phi": np.asarray(PHI_blk[:L], dtype=float),
            "r": np.asarray(R_blk[:L], dtype=float),
            "Q": np.asarray(Q_blk[:L], dtype=float),
            "B": np.asarray(B_blk[:L], dtype=float),
            "attacked": np.asarray(attacked_blk[:L], dtype=int),
        }

        ctx = {
            "U_applied": U,
            "batch": batch,
            "t_fast_start": t_fast_start,
            "n_fast": n_fast,
            # full fast block (useful for replay-history logging)
            "y_fast": np.asarray(Y_blk, dtype=float),
            "x_fast": np.asarray(X_blk, dtype=float),
            "xw_fast": np.asarray(Xw_blk, dtype=float),
            "phi_fast": np.asarray(PHI_blk, dtype=float),
            "u_fast": np.asarray(Ucmd_blk, dtype=float),
            "attacked_fast": np.asarray(attacked_blk, dtype=int),
            "truncated": bool(self.T_fast is not None and self.fast_total >= self.T_fast),
        }

        # pack a minimal StepOut (environment will use ctx["batch"])
        if L > 0:
            y_prev_val = float(Y_blk[L - 2]) if L >= 2 else float(Y_blk[L - 1])
            x_prev_val = float(X_blk[L - 2]) if L >= 2 else float(X_blk[L - 1])
            xw_prev_val = float(Xw_blk[L - 2]) if L >= 2 else float(Xw_blk[L - 1])

            y_prev = np.array([[y_prev_val]], dtype=float)
            y_curr = np.array([[float(Y_blk[L - 1])]], dtype=float)
            y_wowm_curr = np.array([[float(Xw_blk[L - 1])]], dtype=float)

            x_prev = np.array([[x_prev_val]], dtype=float)
            x_curr = np.array([[float(X_blk[L - 1])]], dtype=float)
            x_wowm_curr = np.array([[float(Xw_blk[L - 1])]], dtype=float)

            u_mat = np.array([[float(Ucmd_blk[L - 1])]], dtype=float)
            phi_mat = np.array([[float(PHI_blk[L - 1])]], dtype=float)
            y_hat_mat = np.array([[float(Yhat_blk[L - 1])]], dtype=float)
            Q_mat = np.array([[float(Q_blk[L - 1])]], dtype=float)
            r_mat = np.array([[float(R_blk[L - 1])]], dtype=float)
        else:
            z = np.array([[0.0]], dtype=float)
            y_prev = y_curr = y_wowm_curr = z
            x_prev = x_curr = x_wowm_curr = z
            u_mat = phi_mat = y_hat_mat = Q_mat = r_mat = z

        terminated = bool(getattr(self, "_terminate", False)) or bool(
            self.T_fast is not None and self.fast_total >= self.T_fast
        )

        return StepOut(
            y_prev=y_prev,
            y_curr=y_curr,
            y_wowm_curr=y_wowm_curr,
            x_prev=x_prev,
            x_curr=x_curr,
            x_wowm_curr=x_wowm_curr,
            u=u_mat,
            phi=phi_mat,
            y_hat=y_hat_mat,
            Q=Q_mat,
            r=r_mat,
            ctx=ctx,
            terminated=terminated,
        )
