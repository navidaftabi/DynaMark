
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import chi2, geom


class BetaModelBase:
    """
      - reset(): clears internal history
      - push(U, step_ctx): called once per detector step (stores U and any needed matrices)
      - beta_t(t_det): returns beta_t for the *current* detector time index (1-based)
    """
    def reset(self) -> None:
        raise NotImplementedError

    def push(self, *, U: np.ndarray, step_ctx: Dict[str, Any]) -> None:
        raise NotImplementedError

    def beta_t(self, t_det: int) -> float:
        raise NotImplementedError


@dataclass
class MCBetaConfig:
    """
    Beta model:
      - For all tau in the geometric mixture, we use the same distribution:
            r_t ~ N(0, S_t),
            S_t = Q_t + H_t (U_sum) H_t^T
        where U_sum is either:
          - U_t + U'_t  if U_prime is provided in step_ctx, OR
          - U_t + U_{t-delta} otherwise (delayed replay covariance proxy)
    """
    alpha: float                  # false alarm rate for chi-square detector
    p_geom: float                 # geometric parameter (attack onset prior)
    delta_t: int = 0              # replay delay in detector steps (>=0)
    n_mc: int = 2000              # MC samples for estimating F_t = P(g_t < gamma | attack)
    seed: Optional[int] = None    # RNG seed for reproducibility
    jitter: float = 1e-9          # numerical PSD stabilizer

@dataclass

class BetaLookupConfig:
    """
    Beta model:
        - Precompute beta_t lookup table via Monte Carlo or other method, store in .npz file with keys 't_det' and 'beta_t'.
    """
    lookup_path: str
    clip: bool = True
    eps: float = 1e-12


class ChiSquareBetaMC(BetaModelBase):
    """
    Plant must provide in step_ctx (per step):
      - 'Q': (n,n) detector noise covariance used in g_t = r^T Q^{-1} r
      - 'H': (n,d) Jacobian of predictor wrt watermark input u
            For linear predictor yhat = A y + B u, H = B.
            For nonlinear predictor yhat = g(y,u,...), use the local linearization H = ∂g/∂u
            (constant is fine if your derivations assume it).
      Optional:
      - 'U_prime': (d,d) covariance of replay-side watermark at time t
                   (if you actually model U_t and U'_t separately).
        If absent, we approximate U_prime via delayed U_{t-delta_t}.
    """
    def __init__(self, *, dof: int, u_dim: int, cfg: MCBetaConfig):
        self.n = int(dof)
        self.d = int(u_dim)
        self.cfg = cfg
        self.gamma = float(chi2.ppf(1.0 - cfg.alpha, df=self.n))
        self._rng = np.random.default_rng(cfg.seed)
        self.reset()

    def reset(self) -> None:
        self.U_hist: list[np.ndarray] = []
        self.ctx_hist: list[Dict[str, Any]] = []
        # cache F_t per detector step (1-based key)
        self._F_cache: Dict[int, float] = {}

    def push(self, *, U: np.ndarray, step_ctx: Dict[str, Any]) -> None:
        U = np.asarray(U, dtype=float)
        if U.shape != (self.d, self.d):
            raise ValueError(f"U has shape {U.shape}, expected {(self.d, self.d)}")

        # enforce symmetry
        U = 0.5 * (U + U.T)
        self.U_hist.append(U)
        self.ctx_hist.append(step_ctx)

    def _get_step_mats(self, t_det: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (Q_t, H_t, U_sum_t) for detector step t_det.
        """
        if t_det < 1 or t_det > len(self.U_hist):
            raise IndexError(f"t_det={t_det} out of range for stored history length={len(self.U_hist)}")

        ctx = self.ctx_hist[t_det - 1]

        if "Q" not in ctx:
            raise KeyError("step_ctx must contain 'Q' (detector covariance)")
        Q = np.asarray(ctx["Q"], dtype=float)
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q has shape {Q.shape}, expected {(self.n, self.n)}")

        # Prefer 'H' but allow 'B' as a fallback for linear predictors.
        if "H" in ctx:
            H = np.asarray(ctx["H"], dtype=float)
        elif "B" in ctx:
            H = np.asarray(ctx["B"], dtype=float)
        else:
            raise KeyError("step_ctx must contain 'H' (or 'B' fallback)")

        if H.shape != (self.n, self.d):
            raise ValueError(f"H has shape {H.shape}, expected {(self.n, self.d)}")

        U_t = self.U_hist[t_det - 1]

        if "U_prime" in ctx and ctx["U_prime"] is not None:
            U_prime = np.asarray(ctx["U_prime"], dtype=float)
            if U_prime.shape != (self.d, self.d):
                raise ValueError(f"U_prime has shape {U_prime.shape}, expected {(self.d, self.d)}")
            U_prime = 0.5 * (U_prime + U_prime.T)
        else:
            j = max(1, t_det - int(self.cfg.delta_t)) 
            U_prime = self.U_hist[j - 1]

        U_sum = U_t + U_prime
        # U_sum = 0.5 * (U_sum + U_sum.T)
        return Q, H, U_sum

    def _compute_S(self, t_det: int) -> tuple[np.ndarray, np.ndarray]:
        """
        S_t = Q_t + H_t (U_sum) H_t^T
        Returns (Q_t, S_t).
        """
        Q, H, U_sum = self._get_step_mats(t_det)
        S = Q + H @ U_sum @ H.T
        # numerical safety
        S = 0.5 * (S + S.T) + float(self.cfg.jitter) * np.eye(self.n)
        return Q, S

    # Monte Carlo estimate of F_t
    def _F_t(self, t_det: int) -> float:
        """
        F_t = P( g_t < gamma | attack ), onset-free => independent of tau.
        g_t = r^T Q^{-1} r,  r ~ N(0, S_t).
        """
        if t_det in self._F_cache:
            return self._F_cache[t_det]

        Q, S = self._compute_S(t_det)
        Qinv = np.linalg.inv(Q)

        # Sample r ~ N(0,S) using Cholesky.
        L = np.linalg.cholesky(S)
        Z = self._rng.standard_normal(size=(int(self.cfg.n_mc), self.n))
        R = Z @ L.T  # (n_mc, n)

        # g = r^T Q^{-1} r for each sample
        G = np.einsum("bi,ij,bj->b", R, Qinv, R)
        F = float(np.mean(G < self.gamma))

        self._F_cache[t_det] = F
        return F

    def beta_t(self, t_det: int) -> float:
        # detector time index is treated as 1-based
        if t_det < 1:

            t_det = 1
        F_t = self._F_t(t_det)
        p = float(self.cfg.p_geom)
        # Mixture over onset prior:
        # beta_t = P(no alarm | sigma=1)
        #        = F_t * P(tau <= t) + (1-alpha) * P(tau > t)
        q_tau = float(geom.cdf(t_det, p))
        beta = F_t * q_tau + (1.0 - self.cfg.alpha) * (1.0 - q_tau)
        return float(np.clip(beta, 0.0, 1.0))
    

class BetaLookup(BetaModelBase):
    """
    Offline-calibrated beta lookup for scalar watermark variance U (shape (1,1)).
    Expected .npz payload:
      - U_grid:           shape (K,)
      - beta_hat_t_u:     shape (T, K) or (K, T)
      - g_tilde:          optional scalar
      - tau_grid:         optional
      - tau_weights:      optional
    """
    def __init__(self, *, dof: int, u_dim: int, cfg: BetaLookupConfig):
        self.n = int(dof)
        self.d = int(u_dim)
        self.cfg = cfg
        payload = np.load(cfg.lookup_path, allow_pickle=False)
        self.U_grid = np.asarray(payload["U_grid"], dtype=float).reshape(-1)
        beta_hat = np.asarray(payload["beta_hat_t_u"], dtype=float)
        if beta_hat.ndim != 2:
            raise ValueError("beta_hat_t_u must be a 2D array.")

        # Support either (T, K) or (K, T)
        if beta_hat.shape[1] == self.U_grid.size:
            self.beta_hat_t_u = beta_hat
        elif beta_hat.shape[0] == self.U_grid.size:
            self.beta_hat_t_u = beta_hat.T
        else:
            raise ValueError(
                f"beta_hat_t_u shape {beta_hat.shape} is incompatible with U_grid size {self.U_grid.size}."
            )

        self.T = int(self.beta_hat_t_u.shape[0])
        self.g_tilde = float(payload["g_tilde"]) if "g_tilde" in payload else None
        self.reset()

    def reset(self) -> None:
        self._U_curr_scalar: Optional[float] = None

    def push(self, *, U: np.ndarray, step_ctx: Dict[str, Any]) -> None:
        U = np.asarray(U, dtype=float)
        if U.shape != (self.d, self.d):
            raise ValueError(f"U has shape {U.shape}, expected {(self.d, self.d)}")
        if self.d != 1:
            raise NotImplementedError(
                "BetaLookup currently supports scalar watermark variance only (u_dim=1)."
            )
        self._U_curr_scalar = float(U[0, 0])

    def _interp_beta(self, t_idx: int, u_scalar: float) -> float:
        row = self.beta_hat_t_u[t_idx]  # shape (K,)
        if self.cfg.clip:
            u_scalar = float(np.clip(u_scalar, self.U_grid[0], self.U_grid[-1]))
        if u_scalar <= self.U_grid[0]:
            return float(row[0])
        if u_scalar >= self.U_grid[-1]:
            return float(row[-1])
        j = int(np.searchsorted(self.U_grid, u_scalar, side="right"))
        u_lo, u_hi = float(self.U_grid[j - 1]), float(self.U_grid[j])
        b_lo, b_hi = float(row[j - 1]), float(row[j])
        w_hi = (u_scalar - u_lo) / max(u_hi - u_lo, self.cfg.eps)
        w_lo = 1.0 - w_hi
        return float(w_lo * b_lo + w_hi * b_hi)

    def beta_t(self, t_det: int) -> float:
        if self._U_curr_scalar is None:
            raise RuntimeError("BetaLookup.beta_t() called before push().")

        # current belief filter time can be 0 at the first call; clamp into [1, T]
        t_det = max(1, min(int(t_det), self.T))

        t_idx = t_det - 1
        beta = self._interp_beta(t_idx=t_idx, u_scalar=self._U_curr_scalar)
        return float(np.clip(beta, 0.0, 1.0))
