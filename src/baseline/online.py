from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from .linalg import sym, vecF, solve_psd, regularize_psd
from .opt import opt_trace_ratio


@dataclass
class TACState:
    k: int
    U: np.ndarray               # current exploration covariance (p,p)
    U_star: np.ndarray          # current exploitation covariance (p,p)
    roots: Optional[np.ndarray] # (n_eigs,) complex
    Omega: Optional[np.ndarray] # (m*n_eigs, p) complex
    W_hat: np.ndarray           # (m,m) real
    Uphi_hat: np.ndarray        # (m,m) real (covariance of watermark-induced output component)
    P_hat: np.ndarray           # (p,p) real
    X_hat: np.ndarray           # (p,p) real


class TACOnlineWatermarker:
    """
    Online TAC'18 watermark design + NP detector (Algorithm 1 / Section IV).

    This implementation follows the authors' Julia reference code:
    - online Markov parameter estimation via cross-correlation with watermark input
    - periodic eigenvalue/Omega estimation via trace-based polynomial (Eq. 23–24)
    - watermark covariance update via generalized eigenvector optimization (opt())
    - recursive computation of phihat_k and What and ghat_k

    Controller inclusion (paper Section III-B):
    - Provide u_ctrl (controller command BEFORE watermark) to `step(...)`.
    - The algorithm forms the augmented output y^~ = [y; u_ctrl] internally.
    """

    def __init__(
        self,
        *,
        y_dim: int,
        phi_dim: int,
        n_eigs: int,
        delta: float,
        beta: float = 1/3,
        Xyy: Optional[np.ndarray] = None,
        Xphiphi: Optional[np.ndarray] = None,
        update_interval: int = 100,
        min_k_for_updates: int = 100,
        psd_eps: float = 1e-9,
        rng: Optional[np.random.Generator] = None,
        include_u_dim: int = 0,
    ) -> None:
        if y_dim <= 0:
            raise ValueError("y_dim must be > 0")
        if phi_dim <= 0:
            raise ValueError("phi_dim must be > 0")
        if n_eigs <= 0:
            raise ValueError("n_eigs must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0,1)")
        if delta <= 0:
            raise ValueError("delta must be > 0")
        if update_interval <= 0:
            raise ValueError("update_interval must be > 0")

        self.y_dim = int(y_dim)
        self.u_dim = int(include_u_dim)
        self.m = self.y_dim + self.u_dim
        self.p = int(phi_dim)
        self.n = int(n_eigs)
        self.max_lag = 3 * self.n - 2
        self.ring_len = 3 * self.n - 1
        self.delta = float(delta)
        self.beta = float(beta)
        self.update_interval = int(update_interval)
        self.min_k_for_updates = int(min_k_for_updates)
        self.psd_eps = float(psd_eps)
        self.rng = rng if rng is not None else np.random.default_rng()

        if Xyy is None:
            self.Xyy = np.eye(self.m)
        else:
            Xyy = np.asarray(Xyy, dtype=float)
            if Xyy.shape != (self.m, self.m):
                raise ValueError(f"Xyy must be shape ({self.m},{self.m})")
            self.Xyy = sym(Xyy)

        if Xphiphi is None:
            self.Xphiphi = np.eye(self.p)
        else:
            Xpp = np.asarray(Xphiphi, dtype=float)
            if Xpp.shape != (self.p, self.p):
                raise ValueError(f"Xphiphi must be shape ({self.p},{self.p})")
            self.Xphiphi = sym(Xpp)

        self.k = 0
        self.H_hat = [np.zeros((self.m, self.p), dtype=np.complex128) for _ in range(self.max_lag + 1)]
        self.Y_cov = np.zeros((self.m, self.m), dtype=np.complex128)
        self._phi_pre_ring = np.zeros((self.ring_len, self.p), dtype=np.complex128)
        self.roots: Optional[np.ndarray] = None
        self.Omega: Optional[np.ndarray] = None
        self._hphi = np.zeros((self.m, self.n), dtype=np.complex128)
        self.W_hat = np.eye(self.m, dtype=np.complex128)
        self.Uphi_hat = np.zeros((self.m, self.m), dtype=np.complex128)
        self.P_hat = np.eye(self.p, dtype=np.complex128)
        self.X_hat = np.eye(self.p, dtype=np.complex128)
        self.U_star = opt_trace_ratio(self.P_hat.real, self.X_hat.real, self.delta, eps=self.psd_eps)
        self.U = self.U_star + self.delta * np.eye(self.p)

    def current_U(self) -> np.ndarray:
        """Return current exploration covariance U_k (p x p) used to sample phi_k."""
        return self.U.copy()

    def sample_phi(self) -> np.ndarray:
        """Sample phi_k ~ N(0, U_k)."""
        U = sym(self.U)
        U = regularize_psd(U, eps=self.psd_eps)
        return self.rng.multivariate_normal(mean=np.zeros(self.p), cov=U).reshape(self.p,)

    def get_state(self) -> TACState:
        return TACState(
            k=self.k,
            U=self.U.copy(),
            U_star=self.U_star.copy(),
            roots=None if self.roots is None else self.roots.copy(),
            Omega=None if self.Omega is None else self.Omega.copy(),
            W_hat=sym(self.W_hat.real),
            Uphi_hat=sym(self.Uphi_hat.real),
            P_hat=sym(self.P_hat.real),
            X_hat=sym(self.X_hat.real),
        )
    
    def _roots_ok(self, roots: np.ndarray, *, eps: float = 1e-3, hard_max: float = 1.2) -> bool:
        """Return True if roots look usable/stable."""
        if roots is None:
            return False
        roots = np.asarray(roots)
        if roots.shape != (self.n,):
            return False
        if not np.all(np.isfinite(roots)):
            return False
        max_abs = float(np.max(np.abs(roots)))
        if (not np.isfinite(max_abs)) or (max_abs > hard_max):
            return False
        if np.any(np.abs(roots) >= 1.0 - eps):
            return False
        return True


    def _omega_ok(self, Omega: np.ndarray, *, hard_max: float = 1e6) -> bool:
        if Omega is None:
            return False
        Omega = np.asarray(Omega)
        if Omega.shape != (self.m * self.n, self.p):
            return False
        if not np.all(np.isfinite(Omega)):
            return False
        if float(np.max(np.abs(Omega))) > hard_max:
            return False
        return True

    def step(
        self,
        *,
        y_next: np.ndarray,
        phi_k: np.ndarray,
        U_k: Optional[np.ndarray] = None,
        u_ctrl_k: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Update online estimates using (y_{k+1}, u_k, phi_k) and return outputs.

        Args:
          y_next: measurement AFTER applying phi_k. Shape (y_dim,) or (y_dim,1).
          phi_k:  watermark that was applied at time k. Shape (p,) or (p,1).
          U_k:    covariance used to generate phi_k. If None, uses self.U.
          u_ctrl_k: controller command BEFORE watermark at time k. Shape (u_dim,) if provided.

        Returns dict with keys:
          - U_next: next exploration covariance (p,p) to use for sampling phi_{k+1}
          - U_star: exploitation covariance (p,p)
          - g_hat:  estimated NP statistic (float) or np.nan if not ready
          - resid:  residual vector theta_hat (m,)
          - y_aug:  augmented output vector (m,)
          - roots, Omega, W_hat, Uphi_hat, P_hat, X_hat (diagnostics)
        """
        y_next = np.asarray(y_next, dtype=float).reshape(-1)
        if y_next.size != self.y_dim:
            raise ValueError(f"y_next must have size {self.y_dim}")
        phi_k = np.asarray(phi_k, dtype=float).reshape(-1)
        if phi_k.size != self.p:
            raise ValueError(f"phi_k must have size {self.p}")
        if self.u_dim > 0:
            if u_ctrl_k is None:
                raise ValueError("u_ctrl_k is required because include_u_dim > 0")
            u_ctrl_k = np.asarray(u_ctrl_k, dtype=float).reshape(-1)
            if u_ctrl_k.size != self.u_dim:
                raise ValueError(f"u_ctrl_k must have size {self.u_dim}")
            y_aug = np.concatenate([y_next, u_ctrl_k], axis=0)
        else:
            y_aug = y_next.copy()

        if U_k is None:
            U_used = self.U
        else:
            U_used = np.asarray(U_k, dtype=float)
            if U_used.shape != (self.p, self.p):
                raise ValueError(f"U_k must be shape ({self.p},{self.p})")
            U_used = sym(U_used)

        self.k += 1
        k = self.k
        phi_pre = solve_psd(U_used, phi_k.reshape(self.p, 1)).reshape(self.p,)
        phi_pre = (1.0 / k) * phi_pre
        self._phi_pre_ring[k % self.ring_len, :] = phi_pre.astype(np.complex128)
        y_aug_c = y_aug.astype(np.complex128).reshape(self.m, 1)
        for tau in range(self.max_lag + 1):
            phi_ref = self._phi_pre_ring[(k - tau) % self.ring_len, :].reshape(1, self.p)  # (1,p)
            self.H_hat[tau] = self.H_hat[tau] + y_aug_c @ phi_ref - (self.H_hat[tau] / k)
        self.Y_cov = self.Y_cov + (y_aug_c @ y_aug_c.conj().T - self.Y_cov) / k

        if (k % self.update_interval == 0) and (k >= self.min_k_for_updates):
            old_roots = None if self.roots is None else self.roots.copy()
            old_Omega = None if self.Omega is None else self.Omega.copy()

            self._update_roots_and_omega()

            if (self.roots is None) or (self.Omega is None) or (not self._roots_ok(self.roots)) or (not self._omega_ok(self.Omega)):
                self.roots = old_roots
                self.Omega = old_Omega
            else:
                self._update_Uphi_and_X()

        g_hat = np.nan
        resid = np.full((self.m,), np.nan, dtype=float)

        if self.roots is not None and self.Omega is not None:
            self._update_hphi(phi_k.astype(np.complex128))
            yphi_hat = np.sum(self._hphi, axis=1).reshape(self.m, 1)  # (m,1)
            a = y_aug_c - yphi_hat                                  # residual (m,1)
            resid = a.reshape(-1).real

            # Update W_hat 
            self.W_hat = self.W_hat + (a @ a.conj().T - self.W_hat) / k
            self.W_hat = sym(self.W_hat)

            # Update P_hat 
            self._update_P()

            # Update U_star and U 
            if np.all(np.abs(self.roots) < 1.0 - 1e-6):
                self.U_star = opt_trace_ratio(self.P_hat.real, self.X_hat.real, self.delta, eps=self.psd_eps)
                self.U = self.U_star + (self.delta / (k ** self.beta)) * np.eye(self.p)
                self.U = sym(self.U)
            # else keep previous U

            # Compute estimated NP statistic (paper Eq. 11):
            # g = a^T W^{-1} a - y^T (Uphi + W)^{-1} y
            W = regularize_psd(self.W_hat.real, eps=self.psd_eps)
            UW = regularize_psd((self.Uphi_hat + self.W_hat).real, eps=self.psd_eps)

            term1 = float((a.conj().T @ solve_psd(W, a)).real.squeeze())
            term2 = float((y_aug_c.conj().T @ solve_psd(UW, y_aug_c)).real.squeeze())
            g_hat = term1 - term2

        out = {
            "U_next": self.U.copy(),
            "U_star": self.U_star.copy(),
            "g_hat": float(g_hat) if np.isfinite(g_hat) else np.nan,
            "resid": resid,
            "y_aug": y_aug.copy(),
            "roots": None if self.roots is None else self.roots.copy(),
            "Omega": None if self.Omega is None else self.Omega.copy(),
            "W_hat": sym(self.W_hat.real),
            "Uphi_hat": sym(self.Uphi_hat.real),
            "P_hat": sym(self.P_hat.real),
            "X_hat": sym(self.X_hat.real),
            "k": k,
        }
        return out

    def _update_roots_and_omega(self) -> None:
        """Estimate polynomial coefficients -> roots (eigenvalues) and Omega (Eq. 23–24)."""
        n = self.n
        m = self.m
        p = self.p

        cW1 = np.zeros((m * p * (n + 1), (2 * n - 1)), dtype=np.complex128)
        for i in range(n + 1):
            for j in range(2 * n - 1):
                lag = i + j  # 0..3n-2
                Hij = self.H_hat[lag]
                cW1[i * m * p : (i + 1) * m * p, j] = vecF(Hij)

        mat_k = np.zeros((n, n), dtype=np.complex128)
        trace_vec = np.zeros((n,), dtype=np.complex128)

        def block(i: int, cn: int) -> np.ndarray:
            v = cW1[i * m * p : (i + 1) * m * p, cn]
            return v.reshape((m, p), order="F")

        for i in range(n):
            for j in range(n):
                acc = 0.0 + 0.0j
                for cn in range(2 * n - 1):
                    Hi = block(i, cn)
                    Hj = block(j, cn)
                    acc += np.trace(Hi.conj().T @ Hj)
                mat_k[i, j] = acc

        for i in range(n):
            acc = 0.0 + 0.0j
            for cn in range(2 * n - 1):
                Hi = block(i, cn)
                Hn = block(n, cn)
                acc += np.trace(Hi.conj().T @ Hn)
            trace_vec[i] = acc

        # alpha = -mat_k^{-1} trace_vec
        alpha, *_ = np.linalg.lstsq(mat_k, (-trace_vec).reshape(n, 1), rcond=1e-12)
        alpha = alpha.reshape(n,)
        coeffs_asc = np.concatenate([alpha, np.array([1.0 + 0.0j])])
        roots = np.roots(coeffs_asc[::-1])
        self.roots = roots

        t = np.arange(0, 3 * n - 1, dtype=int)
        V = np.vstack([roots**ti for ti in t])  # (3n-1,n)

        tmp = np.zeros((m * (3 * n - 1), p), dtype=np.complex128)
        for lag in range(0, 3 * n - 1):
            tmp[lag * m : (lag + 1) * m, :] = self.H_hat[lag]

        K = np.kron(V, np.eye(m, dtype=np.complex128))  # (m*(3n-1), m*n)
        Omega, *_ = np.linalg.lstsq(K, tmp, rcond=1e-12)  # (m*n, p)
        self.Omega = Omega

    def _update_Uphi_and_X(self) -> None:
        """Update Uphi_hat (m,m) and X_hat (p,p) using current roots/Omega."""
        assert self.roots is not None and self.Omega is not None
        roots = self.roots
        Omega = self.Omega
        n = self.n
        m = self.m
        p = self.p

        # Compute Uphi_hat = Σ_{i,j} Ω_i U_star Ω_j^H / (1 - r_i r_j^*)
        Uphi = np.zeros((m, m), dtype=np.complex128)
        for i in range(n):
            Oi = Omega[i * m : (i + 1) * m, :]  # (m,p)
            for j in range(n):
                Oj = Omega[j * m : (j + 1) * m, :]
                denom = 1.0 - roots[i] * np.conj(roots[j])
                if np.abs(denom) < 1e-12:
                    continue
                Uphi += (Oi @ self.U_star @ Oj.conj().T) / denom
        self.Uphi_hat = sym(Uphi.real)

        # Compute X_hat = Σ Ω_i^H Xyy Ω_j / (1 - r_i^* r_j) + Xphiphi
        Xhat = np.zeros((p, p), dtype=np.complex128)
        for i in range(n):
            Oi = Omega[i * m : (i + 1) * m, :]
            for j in range(n):
                Oj = Omega[j * m : (j + 1) * m, :]
                denom = 1.0 - np.conj(roots[i]) * roots[j]
                if np.abs(denom) < 1e-12:
                    continue
                Xhat += (Oi.conj().T @ self.Xyy @ Oj) / denom
        Xhat = Xhat + self.Xphiphi
        self.X_hat = sym(Xhat.real)

    def _update_hphi(self, phi_k: np.ndarray) -> None:
        """Update internal filter states hphi[:,i] = r_i hphi[:,i] + Ω_i phi_k."""
        assert self.roots is not None and self.Omega is not None
        roots = self.roots
        Omega = self.Omega
        n = self.n
        m = self.m
        for i in range(n):
            Oi = Omega[i * m : (i + 1) * m, :]  # (m,p)
            self._hphi[:, i] = roots[i] * self._hphi[:, i] + (Oi @ phi_k.reshape(self.p,))

    def _update_P(self) -> None:
        """Update P_hat = Σ Ω_i^H W^{-1} Ω_j / (1 - r_i^* r_j)."""
        assert self.roots is not None and self.Omega is not None
        roots = self.roots
        Omega = self.Omega
        n = self.n
        m = self.m
        p = self.p

        W = regularize_psd(self.W_hat.real, eps=self.psd_eps)
        Phat = np.zeros((p, p), dtype=np.complex128)

        for j in range(n):
            Oj = Omega[j * m : (j + 1) * m, :]  # (m,p)
            W_inv_Oj = solve_psd(W, Oj)          # (m,p)
            for i in range(n):
                Oi = Omega[i * m : (i + 1) * m, :]
                denom = 1.0 - np.conj(roots[i]) * roots[j]
                if np.abs(denom) < 1e-12:
                    continue
                Phat += (Oi.conj().T @ W_inv_Oj) / denom

        self.P_hat = sym(Phat.real)
