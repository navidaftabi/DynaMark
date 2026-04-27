
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from .base import PlantBase
from ..core.types import StepOut
from ..core.utils import _as_2d_mat


class NonlinearMSD:
    """Single MSD, exact slide discretization, with configurable process-noise family."""
    def __init__(
            self,
            m=1.0,
            k1=0.5,
            k2=1.0,
            b1=1.0,
            b2=0.1,
            Ts=0.01,
            Q=None,
            noise_cfg: Optional[Dict[str, Any]] = None,
            seed=0,
    ):
        self.m = float(m)
        self.k1 = float(k1); self.k2 = float(k2)
        self.b1 = float(b1); self.b2 = float(b2)
        self.Ts = float(Ts)
        self.Q = np.array(Q if Q is not None else np.diag([1e-6, 1e-6]), dtype=float)
        self.noise_cfg = dict(noise_cfg or {"family": "gaussian"})
        self.rng = np.random.default_rng(seed)

        if np.allclose(self.Q, np.diag(np.diag(self.Q))):
            self._diag_std = np.sqrt(np.diag(self.Q))
            self._chol = None
        else:
            self._diag_std = None
            self._chol = np.linalg.cholesky(self.Q + 1e-15 * np.eye(self.Q.shape[0]))
        
        # for Laplace and contaminated-Gaussian, use the diagonal of Q as target variance
        self._diag_var = np.diag(self.Q).astype(float)
        self._dim = int(self.Q.shape[0])

    def b(self, v):
        return self.b1 * v + self.b2 * (v ** 3)

    def k(self, p):
        return self.k1 * p + self.k2 * (p ** 3)

    def g(self, y, u, F):
        # y is shape (2,) in this internal function
        p, v = float(y[0]), float(y[1])
        p_next = p + self.Ts * v
        v_next = v - (self.Ts/self.m) * ( self.b(v) + self.k(p) - u - F )
        return np.array([p_next, v_next], dtype=float)
    
    def _sample_gaussian(self) -> np.ndarray:
        if self._diag_std is not None:
            return self.rng.normal(loc=0.0, scale=self._diag_std, size=self._dim)
        z = self.rng.normal(size=self._dim)
        return self._chol @ z
    
    def _sample_student_t(self, df: float) -> np.ndarray:
        if df <= 2:
            raise ValueError("student_t noise requires df > 2 for finite variance.")
        # scale so that Cov[w] = Q
        scale = np.sqrt((df - 2.0) / df)
        if self._diag_std is not None:
            z = self.rng.standard_t(df=df, size=self._dim)
            return scale * self._diag_std * z
        z = self.rng.standard_t(df=df, size=self._dim)
        return scale * (self._chol @ z)
    
    def _sample_laplace(self) -> np.ndarray:
        # coordinate-wise Laplace matched to diag(Q): Var = 2 b^2
        # if Q is not diagonal, we still use the diagonal variances as a pragmatic robustness test.
        b = np.sqrt(np.maximum(self._diag_var, 0.0) / 2.0)
        return self.rng.laplace(loc=0.0, scale=b, size=self._dim)
    
    def _sample_contaminated_gaussian(self, p: float, kappa: float) -> np.ndarray:
        if not (0.0 <= p < 1.0):
            raise ValueError("contaminated_gaussian requires 0 <= p < 1.")
        if kappa <= 0.0:
            raise ValueError("contaminated_gaussian requires kappa > 0.")
        # choose base covariance so total covariance matches Q in expectation:
        # (1-p) Q_base + p * kappa * Q_base = Q
        denom = (1.0 - p) + p * kappa
        if denom <= 0.0:
            raise ValueError("Invalid contaminated-Gaussian parameters.")
        if self._diag_std is not None:
            base_std = self._diag_std / np.sqrt(denom)
            burst = self.rng.uniform() < p
            scale = np.sqrt(kappa) if burst else 1.0
            return self.rng.normal(loc=0.0, scale=scale * base_std, size=self._dim)
        base_chol = self._chol / np.sqrt(denom)
        burst = self.rng.uniform() < p
        scale = np.sqrt(kappa) if burst else 1.0
        z = self.rng.normal(size=self._dim)
        return scale * (base_chol @ z)
    
    def sample_w(self) -> np.ndarray:
        family = str(self.noise_cfg.get("family", "gaussian")).lower()
        if family == "gaussian":
            return self._sample_gaussian()
        if family == "student_t":
            df = float(self.noise_cfg.get("df", 5.0))
            return self._sample_student_t(df=df)
        if family == "laplace":
            return self._sample_laplace()
        if family == "contaminated_gaussian":
            p = float(self.noise_cfg.get("p", 0.05))
            kappa = float(self.noise_cfg.get("kappa", 10.0))
            return self._sample_contaminated_gaussian(p=p, kappa=kappa)
        raise ValueError(f"Unsupported MSD noise family: {family}")

    def step(self, y, u, F):
        y_det = self.g(y, u, F)
        w = self.sample_w()
        return y_det + w, w
    
def solve_equilibrium_p(k1, k2, F):
    """Solve k(p)=F => k2 p^3 + k1 p - F = 0 and pick a real root."""
    roots = np.roots([k2, 0.0, k1, -F])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    if not real_roots:
        raise RuntimeError("No real equilibrium root found.")
    return float(sorted(real_roots, key=lambda x: abs(x))[0])


class MSDNonlinearPlant(PlantBase):
    y_dim = 2
    u_dim = 1

    def __init__(self, data: Dict[str, Any], *, seed: Optional[int] = None):
        self.data = data
        self._seed = int(0 if seed is None else seed)
        self._rng = np.random.default_rng(self._seed)

        self.Ts = float(data.get("Ts", 0.01))
        self.T = int(data.get("T", 1000))

        # MSD params
        self.m = float(data.get("m", 1.0))
        self.k1 = float(data.get("k1", 0.5))
        self.k2 = float(data.get("k2", 1.0))
        self.b1 = float(data.get("b1", 1.0))
        self.b2 = float(data.get("b2", 0.1))

        Q = data.get("Q", np.diag([1e-6, 1e-6]).tolist())
        self.Q = _as_2d_mat(Q, name="Q")

        # process-noise model (default remains Gaussian)
        self.noise_cfg = dict(data.get("noise", {"family": "gaussian"}))

        # exogenous
        self.F0 = float(data.get("F0", 0.0))

        # initial condition (true + observed start identical)
        p_eq = solve_equilibrium_p(self.k1, self.k2, self.F0)
        self._y0 = np.asarray([p_eq, 0.0], dtype=float).reshape(2)

        # u_star chirp params 
        chirp = data.get("u_star", {})
        self.u_amp = float(chirp.get("amp", 0.1))
        self.w0 = float(chirp.get("w0", 0.1))
        self.wf = float(1.0 + self.F0)

        # disturbance config
        dist = data.get("disturbance", {"type": "constant"})
        self.dist_type = str(dist.get("type", "constant")).lower()
        self.dist_params = dist

        # system
        self.sys = NonlinearMSD(
            m=self.m, 
            k1=self.k1, k2=self.k2, 
            b1=self.b1, b2=self.b2,
            Ts=self.Ts, Q=self.Q, 
            noise_cfg=self.noise_cfg,
            seed=self._seed
        )

        # Jacobian H = ∂g/∂u = [0; Ts/m] 
        self.H = np.array([[0.0], [self.Ts / self.m]], dtype=float)  # (2,1)

        # internal states
        self.reset(seed=self._seed)

    @property
    def y_curr(self) -> np.ndarray:
        # current OBSERVED measurement (what estimator sees)
        return self._y.reshape(2, 1)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

        # Rebuild the system so its internal RNG is reset too.
        self.sys = NonlinearMSD(
            m=self.m,
            k1=self.k1, k2=self.k2,
            b1=self.b1, b2=self.b2,
            Ts=self.Ts, Q=self.Q,
            noise_cfg=self.noise_cfg,
            seed=self._seed,
        )

        self._t = 0
        # true and observed start the same
        self._y = self._y0.copy()  # (2,)
        self._x = self._y0.copy()  # (2,)

        # wowm reference (no watermark)
        self._y_wowm = self._y0.copy()  # (2,)
        self._x_wowm = self._y0.copy()  # (2,)

    # ---------- exogenous signals ----------
    def _disturbance_F(self, t: float) -> float:
        if self.dist_type == "constant":
            return float(self.dist_params.get("F0", self.F0))
        if self.dist_type == "sin":
            A = float(self.dist_params.get("amp", 0.0))
            w = float(self.dist_params.get("omega", 1.0))
            return float(self.F0 + A * np.sin(w * t))
        return float(self.F0)

    def _u_star(self, t: float, Tf: float) -> float:
        if Tf <= 0:
            return 0.0
        win = self.w0 * (self.wf / self.w0) ** (t / Tf)
        return float(self.u_amp * np.sin(win * t))

    def _sample_phi(self, U: np.ndarray) -> np.ndarray:
        # scalar cov (1,1) -> scalar normal
        U_scalar = float(np.asarray(U, dtype=float).reshape(1, 1)[0, 0])
        if U_scalar < 0:
            raise ValueError("U must be >= 0 for scalar watermark covariance")
        std = float(np.sqrt(U_scalar))
        return np.array([[self._rng.normal(loc=0.0, scale=std)]], dtype=float)  # (1,1)

    # ---------- main step ----------
    def step(self, U: np.ndarray, *, next_y_override=None) -> StepOut:
        """
        One detector step (t -> t+1).
        """
        U = np.asarray(U, dtype=float)
        if U.shape != (1, 1):
            raise ValueError(f"MSD expects U shape (1,1), got {U.shape}")

        # time
        t0 = float(self._t * self.Ts)
        Tf = float(self.T * self.Ts)

        # current observed and true (columns for StepOut)
        y_prev = self._y.reshape(2, 1)        # OBSERVED y_t (replayed if attack already active)
        x_prev = self._x.reshape(2, 1)       # TRUE x_t (physical)

        # wowm previous
        y_wowm_prev = self._y_wowm.reshape(2, 1)

        # exogenous
        F = self._disturbance_F(t0)
        u_star = self._u_star(t0, Tf)

        # watermark + commanded control
        phi = self._sample_phi(U)                             # (1,1)
        u_cmd = np.array([[u_star]], dtype=float) + phi       # (1,1)

        # predictor uses OBSERVED y_t
        y_hat = self.sys.g(y_prev, float(u_cmd.item()), F).reshape(2, 1)

        # true evolution uses flipped control if attack is active this step
        if next_y_override is None:
            u_applied = float(u_cmd.item())
        else:
            u_applied = -float(u_cmd.item())  # flip control under replay attack

        x_next_vec, w = self.sys.step(self._x, u_applied, F)  # both (2,)
        x_next = x_next_vec.reshape(2, 1)

        # observed next measurement
        if next_y_override is None:
            y_obs_next = x_next.copy()
        else:
            y_obs_next = np.asarray(next_y_override, dtype=float).reshape(2, 1)

        # wowm evolution (no watermark), same noise w for fair comparison
        u_wowm = float(u_star)
        y_wowm_det = self.sys.g(y_wowm_prev, u_wowm, F)
        y_wowm_next = (y_wowm_det + w).reshape(2, 1)

        # residual uses OBSERVED y_{t+1}
        r = y_obs_next - y_hat

        # advance internal states
        self._t += 1
        self._x = x_next_vec.reshape(2)       # true physical state
        self._y = y_obs_next.reshape(2)             # observed (possibly replayed)
        self._y_wowm = y_wowm_next.reshape(2)           # reference

        terminated = bool(self._t >= self.T)

        ctx = {
            "Q": self.Q,   # detector covariance
            "H": self.H,   # ∂g/∂u for beta model
            "t": self._t,
            "Ts": self.Ts,
            "m": self.m,
        }

        # StepOut expects x_curr and x_wowm_curr as columns
        x_curr = self._x.reshape(2, 1)
        x_wowm_curr = self._y_wowm.reshape(2, 1)

        return StepOut(
            y_prev=y_prev,               # observed y_t
            y_curr=y_obs_next,           # observed y_{t+1}
            y_wowm_curr=y_wowm_next,
            x_prev=x_prev,               # true x_t
            x_curr=x_curr,               # true x_{t+1}
            x_wowm_curr=x_wowm_curr,
            u=u_cmd,                     # commanded control (before flip)
            phi=phi,
            y_hat=y_hat,
            Q=self.Q,
            r=r,
            ctx=ctx,
            terminated=terminated,
        )
