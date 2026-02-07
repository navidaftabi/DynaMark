
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from .base import PlantBase
from ..core.types import StepOut
from ..core.utils import (
    _as_2d_mat,
    _symmetrize_psd
)

class DigitalTwinLTIPlant(PlantBase):
    """
    Digital Twin linear plant.

    Dynamics:
      x_{t+1} = A x_t + B u_applied + w_t
      y_t = x_t   (full-state measurement)

    P-Controller (no watermark):
      u*_t = Kp (y_bar - y_t)

    Watermarked control:
      u_t = u*_t + phi_t,    phi_t ~ N(0, U_t)

    Wowm reference trajectory (for reward term):
      x^{wowm}_{t+1} = A x^{wowm}_t + B u^{wowm}_t + w_t
      u^{wowm}_t = Kp (y_bar - y^{wowm}_t)

    Replay hook:
      - If next_y_override is provided, we set y_{t+1} := next_y_override
      - AND apply flip by u_attack = -2 u_t  => u_applied = -u_t
    """

    def __init__(self, data: Dict[str, Any], *, seed: Optional[int] = None):
        self.A = _as_2d_mat(data.get("A"), name="A")
        self.B = _as_2d_mat(data.get("B"), name="B")
        self.Kp = _as_2d_mat(data.get("Kp"), name="Kp")
        self.y_bar = _as_2d_mat(data.get("y_bar"), name="y_bar")
        self.mu0 = _as_2d_mat(data.get("mu0"), name="mu0")

        self.Sigma = _symmetrize_psd(_as_2d_mat(data.get("Sigma"), name="Sigma"), name="Sigma")
        self.Q = _symmetrize_psd(_as_2d_mat(data.get("Q"), name="Q"), name="Q")

        self.n = int(self.A.shape[0])
        self.d = int(self.B.shape[1])

        # dims expected by DynaMarkEnv
        self.y_dim = self.n
        self.u_dim = self.d

        # horizon (optional)
        self.T = int(data.get("T")) if data.get("T") is not None else None

        # rng
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # state holders
        self._t = 0
        self._x = None
        self._y = None
        self._x_wowm = None
        self._y_wowm = None
        self._check_shapes()

    def _check_shapes(self) -> None:
        if self.A.shape != (self.n, self.n):
            raise ValueError(f"A must be (n,n), got {self.A.shape}")
        if self.B.shape[0] != self.n:
            raise ValueError(f"B must be (n,d), got {self.B.shape}")
        if self.Kp.shape[1] != self.n:
            raise ValueError(f"Kp must be (d,n), got {self.Kp.shape}")
        if self.Kp.shape[0] != self.B.shape[1]:
            raise ValueError(f"Kp rows must match control dim d={self.d}, got {self.Kp.shape}")
        if self.y_bar.shape not in [(self.n, 1), (self.n,)]:
            raise ValueError(f"y_bar must be (n,1) or (n,), got {self.y_bar.shape}")
        if self.mu0.shape not in [(self.n, 1), (self.n,)]:
            raise ValueError(f"mu0 must be (n,1) or (n,), got {self.mu0.shape}")
        if self.Sigma.shape != (self.n, self.n):
            raise ValueError(f"Sigma must be (n,n), got {self.Sigma.shape}")
        if self.Q.shape != (self.n, self.n):
            raise ValueError(f"Q must be (n,n), got {self.Q.shape}")

    @property
    def y_curr(self) -> np.ndarray:
        # current measurement y_t (column vector)
        return self._y

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self._t = 0
        # Sample initial state x0 ~ N(mu0, Sigma)
        mu = self.mu0.reshape(-1)
        Sigma = self.Sigma
        x0 = _as_2d_mat(self._rng.multivariate_normal(mean=mu, cov=Sigma), name="x0")

        self._x = x0.copy()
        self._y = x0.copy()
        # wowm reference starts from same x0
        self._x_wowm = x0.copy()
        self._y_wowm = x0.copy()

    def _sample_process_noise(self) -> np.ndarray:
        # w_t ~ N(0, Q)
        Q = self.Q
        return _as_2d_mat(self._rng.multivariate_normal(mean=np.zeros(self.n), cov=Q), name="_sample_process_noise")

    def _sample_watermark(self, U: np.ndarray) -> np.ndarray:
        return _as_2d_mat(self._rng.multivariate_normal(mean=np.zeros(self.d), cov=U), name="_sample_watermark")

    def step(self, U: np.ndarray, *, next_y_override=None) -> StepOut:
        """
        Returns StepOut with:
          - y_prev: y_t
          - y_curr: y_{t+1} (possibly replayed if override is given)
          - y_wowm_curr: y^{wowm}_{t+1}
          - y_hat: A y_t + B u_t  (predictor used for residual)
          - r: y_curr - y_hat
          - Q: detector covariance (n,n)
          - ctx: includes Q and H (=B), plus A/B/Kp for debugging
        """
        U = np.asarray(U, dtype=float)
        if U.shape != (self.d, self.d):
            raise ValueError(f"U must be ({self.d},{self.d}), got {U.shape}")

        y_prev = self._y.copy()
        x_prev = self._x.copy()

        # same noise drives wowm + real
        w_t = self._sample_process_noise()

        # watermark
        phi_t = self._sample_watermark(U)

        # baseline controls
        u_wowm = self.Kp @ (self.y_bar - self._y_wowm)
        u = (self.Kp @ (self.y_bar - self._y)) + phi_t

        # wowm evolution
        x_wowm_next = (self.A @ self._x_wowm) + (self.B @ u_wowm) + w_t
        y_wowm_next = x_wowm_next.copy()

        # predictor (for residual)
        y_hat = (self.A @ y_prev) + (self.B @ u)

        # real evolution + replay hook
        if next_y_override is None:
            x_next = (self.A @ x_prev) + (self.B @ u) + w_t
            y_next = x_next.copy()
        else:
            # exact physical attack from your old code: u_attack = -2u => applied = -u
            u_applied = -u
            x_next = (self.A @ x_prev) + (self.B @ u_applied) + w_t
            y_next = _as_2d_mat(next_y_override, name="next_y_override")

        r = y_next - y_hat

        # update internal state
        self._t += 1
        self._x = x_next
        self._y = y_next
        self._x_wowm = x_wowm_next
        self._y_wowm = y_wowm_next

        terminated = False
        if self.T is not None:
            terminated = (self._t >= self.T)

        ctx = {
            "Q": self.Q,
            "H": self.B,   # for linear predictor yhat = A y + B u, H = ∂yhat/∂u = B
            "A": self.A,
            "B": self.B,
            "Kp": self.Kp,
            "t": self._t,
        }

        return StepOut(
            y_prev=y_prev,
            y_curr=y_next,
            y_wowm_curr=y_wowm_next,
            x_prev=x_prev,
            x_curr=x_next,
            x_wowm_curr=x_wowm_next,
            u=u,
            phi=phi_t,
            y_hat=y_hat,
            r=r,
            Q=self.Q,
            ctx=ctx,
            terminated=terminated,
        )
