import numpy as np

def cholesky_action_dim(d: int) -> int:
    return d * (d + 1) // 2

def unpack_cholesky(action: np.ndarray, d: int) -> np.ndarray:
    """
    action = [L11..Ldd, L21,L31,L32,...]  (diagonal first, then strict-lower row-wise)
    Diagonal is assumed >= 0 by action_space bounds.
    Returns L (dxd) lower-triangular.
    """
    a = np.asarray(action, dtype=float).ravel()
    need = cholesky_action_dim(d)
    if a.size != need:
        raise ValueError(f"Bad action size {a.size}, expected {need} for d={d}")

    L = np.zeros((d, d), dtype=float)
    diag = a[:d]
    L[np.arange(d), np.arange(d)] = diag

    idx = d
    for i in range(1, d):
        for j in range(0, i):
            L[i, j] = a[idx]
            idx += 1
    return L

def cov_from_action(action: np.ndarray, d: int, eps: float = 1e-40) -> np.ndarray:
    L = unpack_cholesky(action, d)
    U = L @ L.T
    # numerical safety
    U = 0.5 * (U + U.T)
    # U.flat[:: d + 1] += eps
    return U
