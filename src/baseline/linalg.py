from __future__ import annotations

import numpy as np
import scipy.linalg


def sym(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T)/2 (or Hermitian sym for complex)."""
    return (A + A.conj().T) / 2.0


def vecF(M: np.ndarray) -> np.ndarray:
    """Column-major vectorization (Julia/Matlab style)."""
    return np.asarray(M, dtype=np.complex128).reshape(-1, order="F")


def solve_psd(A: np.ndarray, B: np.ndarray, *, rcond: float = 1e-12) -> np.ndarray:
    """Solve A X = B for symmetric PSD A."""
    A = np.asarray(A)
    B = np.asarray(B)
    reg = 0.0
    for _ in range(3):
        try:
            L = np.linalg.cholesky(A + reg * np.eye(A.shape[0], dtype=A.dtype))
            # Solve L L^H X = B
            Y = scipy.linalg.solve_triangular(L, B, lower=True, overwrite_b=False, check_finite=False)
            X = scipy.linalg.solve_triangular(L.conj().T, Y, lower=False, overwrite_b=False, check_finite=False)
            return X
        except np.linalg.LinAlgError:
            reg = 1e-12 if reg == 0.0 else reg * 10.0
    X, *_ = np.linalg.lstsq(A, B, rcond=rcond)
    return X


def regularize_psd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    A = sym(np.asarray(A))
    w, V = np.linalg.eigh(A.real if np.isrealobj(A) else A)
    w = np.real(w)
    w_clipped = np.maximum(w, eps)
    return (V * w_clipped) @ V.conj().T


def discrete_lyapunov_kron(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Solve Y = A Y A^T + X via Kronecker.
    """
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    n = A.shape[0]
    I = np.eye(n*n, dtype=float)
    K = np.kron(A, A)
    y = np.linalg.solve(I - K, X.reshape(-1, order="F"))
    return y.reshape(n, n, order="F")
