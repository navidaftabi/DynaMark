from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
from collections import deque


@dataclass
class Transition:
    obs: np.ndarray
    act: np.ndarray
    rew: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, obs, act, rew, next_obs, done) -> None:
        self._buf.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                act=np.asarray(act, dtype=np.float32),
                rew=float(rew),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = len(self._buf)
        if n < batch_size:
            raise ValueError(f"Not enough samples: have {n}, need {batch_size}")
        idx = rng.choice(n, size=int(batch_size), replace=False)
        batch = [self._buf[i] for i in idx]

        obs = np.stack([b.obs for b in batch], axis=0)
        act = np.stack([b.act for b in batch], axis=0)
        rew = np.asarray([b.rew for b in batch], dtype=np.float32)
        next_obs = np.stack([b.next_obs for b in batch], axis=0)
        done = np.asarray([b.done for b in batch], dtype=np.float32)
        return obs, act, rew, next_obs, done
