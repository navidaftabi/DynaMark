
import numpy as np
from typing import Any, Dict

class GMM:
    """
    Lightweight 1D Gaussian mixture sampler (no sklearn).
    JSON per segment:
      weights: (K,)
      means:   (K,)
      covs/vars/variances: (K,)  (variances)
    """
    def __init__(self, cfg: Dict[str, Any], rng: np.random.Generator):
        w = np.asarray(cfg["weights"], dtype=float).reshape(-1)
        m = np.asarray(cfg["means"], dtype=float).reshape(-1)
        v = np.asarray(cfg.get("vars", cfg.get("covs", cfg.get("variances"))), dtype=float).reshape(-1)

        if not (w.size == m.size == v.size):
            raise ValueError("GMM params sizes mismatch")
        if np.any(v < 0):
            raise ValueError("GMM variances must be >= 0")
        s = float(np.sum(w))
        if s <= 0:
            raise ValueError("GMM weights must sum to > 0")

        self.w = w / s
        self.m = m
        self.std = np.sqrt(v + 1e-15)
        self.rng = rng

    def sample(self) -> float:
        k = int(self.rng.choice(len(self.w), p=self.w))
        return float(self.rng.normal(loc=self.m[k], scale=self.std[k]))
    
def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))

def _as_array(x: Any, *, dtype=float) -> np.ndarray:
    """Convert JSON-loaded lists/scalars to numpy array."""
    arr = np.array(x, dtype=dtype)
    return arr

def _to_1d(arr: Any, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size < 1:
        raise ValueError(f"{name} is empty")
    return a


def _as_2d_mat(x: Any, *, name: str) -> np.ndarray:
    """
    Ensure x is a 2D matrix ndarray.
    Accepts:
      - nested lists (already 2D)
      - flat list (interpreted as column vector)
      - scalar (1x1)
    """
    arr = _as_array(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"{name} must be 0D/1D/2D, got shape {arr.shape}")
    return arr


def _symmetrize_psd(M: np.ndarray, *, jitter: float = 1e-10, name: str = "M") -> np.ndarray:
    """Symmetrize and add small jitter to diagonal to avoid numerical issues."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got {M.shape}")
    M.flat[:: M.shape[0] + 1] += jitter
    return M