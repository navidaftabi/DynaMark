from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConstantPolicy:
    """
    Deterministic policy returning a constant action vector (length = act_dim).
    """
    action: np.ndarray

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, t: int = 0) -> np.ndarray:
        return np.asarray(self.action, dtype=np.float32).copy()


class ZeroActionPolicy(ConstantPolicy):
    def __init__(self, act_dim: int):
        super().__init__(action=np.zeros((int(act_dim),), dtype=np.float32))

class ConstantActionPolicy(ConstantPolicy):
    """
    This policy returns a constant Cholesky action vector L in U=LL^T.
    Input:
       - constant covariance matrix U (np.ndarray, shape (d,d))
    """
    def __init__(self, U: np.ndarray):
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError("Input U must be a square matrix.")
        d = U.shape[0]
        L = np.linalg.cholesky(U)
        action = []
        for i in range(d):
            for j in range(i + 1):
                action.append(L[i, j])
        action = np.asarray(action, dtype=np.float32)
        super().__init__(action=action)
