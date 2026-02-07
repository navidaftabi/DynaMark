from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


def chol_action_dim(d: int) -> int:
    return int(d * (d + 1) // 2)


@dataclass(frozen=True)
class ActionPackSpec:
    """How action vector is packed: [diag(d), offdiag(d(d-1)/2)]"""
    d: int

    @property
    def act_dim(self) -> int:
        return chol_action_dim(self.d)

    @property
    def off_dim(self) -> int:
        return int(self.d * (self.d - 1) // 2)


class ActorCholesky(nn.Module):
    """
    Actor network that outputs a Cholesky-parameter action vector of length d(d+1)/2.
    First d entries correspond to diagonals (>= 0).
    Remaining entries correspond to off-diagonals (unconstrained).
    """

    def __init__(self, obs_dim: int, d: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.d = int(d)
        self.spec = ActionPackSpec(d=self.d)

        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.diag_head = nn.Linear(hidden_dim, self.d)
        if self.spec.off_dim > 0:
            self.off_head = nn.Linear(hidden_dim, self.spec.off_dim)
        else:
            self.off_head = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns packed action tensor: shape (B, act_dim)
        """
        x = F.leaky_relu(self.norm1(self.fc1(obs)))
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = F.leaky_relu(self.norm3(self.fc3(x)))

        diag = torch.abs(self.diag_head(x))

        if self.off_head is not None:
            off = self.off_head(x)
            act = torch.cat([diag, off], dim=-1)
        else:
            act = diag
        return act


class CriticQ(nn.Module):
    """Q(s,a) critic"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.fc1 = nn.Linear(self.obs_dim + self.act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = F.leaky_relu(self.norm1(self.fc1(x)))
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        return self.q(x).squeeze(-1)
