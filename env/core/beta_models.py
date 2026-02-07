
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
          - U_t + U'_t  if you provide U_prime in step_ctx, OR
          - U_t + U_{t-delta} otherwise (delayed replay covariance proxy)
    """
    alpha: float                  # false alarm rate for chi-square detector
    p_geom: float                 # geometric parameter (attack onset prior)
    window_size: Optional[int]    # None => use all tau from 0..t; else last window
    delta_t: int = 0              # replay delay in detector steps (>=0)
    n_mc: int = 2000              # MC samples for estimating F_t = P(g_t < gamma | attack)
    seed: Optional[int] = None    # RNG seed for reproducibility
    jitter: float = 1e-9          # numerical PSD stabilizer


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
        S_t = Q_t + H_t (U_sum) H_t^T   (vector/matrix-safe)
        Returns (Q_t, S_t).
        """
        Q, H, U_sum = self._get_step_mats(t_det)

        S = Q + H @ U_sum @ H.T
        # numerical safety
        S = 0.5 * (S + S.T)
        S.flat[:: self.n + 1] += float(self.cfg.jitter)

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
        """
        beta_t = P(I_t=0 | Sigma=1) under geometric onset mixture, using (m,S).
          beta = chi2.cdf(gamma; n) * (1 - GeomCDF(t))
               + sum_{tau=start..t} F_{t,tau} * GeomPMF(tau)
        """
        if self.cfg.window_size is None:
            start_tau = 1
        else:
            start_tau = max(1, t_det - int(self.cfg.window_size) + 1)

        base = float(chi2.cdf(self.gamma, df=self.n) * (1.0 - geom.cdf(t_det, self.cfg.p_geom)))

        if start_tau <= 1:
            mass = float(geom.cdf(t_det, self.cfg.p_geom))
        else:
            mass = float(geom.cdf(t_det, self.cfg.p_geom) - geom.cdf(start_tau - 1, self.cfg.p_geom))

        F = self._F_t(t_det)
        return float(base + F * mass)
