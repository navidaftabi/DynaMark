from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OUNoise:
    dim: int
    mu: float = 0.0
    theta: float = 0.15
    sigma_start: float = 0.2
    sigma_decay: float = 0.995
    sigma_min: float = 0.05

    def __post_init__(self):
        self.dim = int(self.dim)
        self.sigma = float(self.sigma_start)
        self.state = np.ones((self.dim,), dtype=np.float32) * float(self.mu)

    def reset(self):
        self.state = np.ones((self.dim,), dtype=np.float32) * float(self.mu)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * rng.standard_normal(self.dim).astype(np.float32)
        self.state = (self.state + dx).astype(np.float32)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return self.state.copy()
