from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


ArrayLike = Union[float, int, list, np.ndarray]


def _as_square_cov(x: ArrayLike, d: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)

    # Allow scalar U for any d as isotropic covariance U I.
    if arr.ndim == 0 or arr.size == 1:
        val = float(arr.reshape(-1)[0])
        if val < 0:
            raise ValueError(f"{name} must be nonnegative.")
        return val * np.eye(d, dtype=float)

    if arr.shape != (d, d):
        raise ValueError(f"{name} must be scalar or shape {(d, d)}, got {arr.shape}.")

    arr = 0.5 * (arr + arr.T)
    eigvals = np.linalg.eigvalsh(arr)
    if eigvals.min() < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite.")

    return arr


def _pack_cholesky_action(U: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """
    Pack U as the action vector expected by env.core.covariance.cov_from_action:
    [diag(L), strict-lower(L) row-wise], where U = L L^T.
    """
    U = np.asarray(U, dtype=float)
    U = 0.5 * (U + U.T)
    d = U.shape[0]

    if np.allclose(U, 0.0):
        L = np.zeros_like(U)
    else:
        try:
            L = np.linalg.cholesky(U)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(U + jitter * np.eye(d))

    action = []

    # First d entries are the Cholesky diagonals.
    for i in range(d):
        action.append(L[i, i])

    # Remaining entries are strict-lower row-wise.
    for i in range(1, d):
        for j in range(i):
            action.append(L[i, j])

    return np.asarray(action, dtype=np.float32)


@dataclass
class BeliefAdaptiveLinearPolicy:
    """
    Non-RL belief adaptive watermark baseline:

        U_t = U_min + (U_max - U_min) d_t,

    where d_t is the detector belief stored as the last entry of the observation.
    """

    U_min: np.ndarray
    U_max: np.ndarray
    d_index: int = -1
    d_clip_min: float = 0.0
    d_clip_max: float = 1.0

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, t: int = 0) -> np.ndarray:
        obs = np.asarray(obs, dtype=float).reshape(-1)

        d_t = float(obs[self.d_index])
        d_t = float(np.clip(d_t, self.d_clip_min, self.d_clip_max))

        U_t = self.U_min + (self.U_max - self.U_min) * d_t
        U_t = 0.5 * (U_t + U_t.T)

        return _pack_cholesky_action(U_t)