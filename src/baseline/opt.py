from __future__ import annotations

import numpy as np
import scipy.linalg

from .linalg import sym, regularize_psd


def opt_trace_ratio(cP: np.ndarray, 
                    cX: np.ndarray, 
                    delta: float, *, eps: float = 1e-9) -> np.ndarray:
    """
    Solve: max tr(U cP) s.t. tr(U cX) <= delta, U >= 0.

    The TAC paper shows an optimal solution of (typically) rank-1:
        U* = delta * z z^T / (z^T cX z)
    where z is the generalized eigenvector of (cP, cX) with the largest |eig|.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")

    cP = sym(np.asarray(cP, dtype=float))
    cX = sym(np.asarray(cX, dtype=float))
    cXr = regularize_psd(cX, eps=eps)
    vals, vecs = scipy.linalg.eigh(cP, cXr, check_finite=False)
    idx = int(np.argmax(np.abs(vals)))
    z = vecs[:, idx:idx+1]  # (p,1)

    denom = float((z.T @ cXr @ z).squeeze())
    if denom <= 0:
        p = cP.shape[0]
        return (delta / np.trace(cXr)) * np.eye(p)

    U = (delta / denom) * (z @ z.T)
    U = sym(U)
    return U
