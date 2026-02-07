from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.policies.networks import ActorCholesky, CriticQ
from src.agents.replay_buffer import ReplayBuffer
from src.agents.noise import OUNoise


@dataclass
class DDPGConfig:
    hidden_dim: int = 128
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 5e-3

    buffer_size: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 5_000

    # exploration
    eps_start: float = 0.2          # prob of random action
    eps_end: float = 0.02
    eps_decay: float = 0.9995

    # target/scheduler cadence
    target_update_every: int = 1
    grad_clip: float = 1.0


class DDPGAgent:
    """
    DDPG with:
      - Actor outputs Cholesky-action params (diag>=0)
      - Twin critics
      - Target networks
      - OU exploration noise
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        d: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: DDPGConfig,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.d = int(d)
        self.act_dim = int(len(action_low))

        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(self.act_dim)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(self.act_dim)

        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))

        # networks
        self.actor = ActorCholesky(self.obs_dim, self.d, hidden_dim=cfg.hidden_dim).to(self.device)
        self.actor_t = ActorCholesky(self.obs_dim, self.d, hidden_dim=cfg.hidden_dim).to(self.device)

        self.critic1 = CriticQ(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.critic2 = CriticQ(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.critic1_t = CriticQ(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.critic2_t = CriticQ(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self._hard_update(self.actor_t, self.actor)
        self._hard_update(self.critic1_t, self.critic1)
        self._hard_update(self.critic2_t, self.critic2)

        # optim
        self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_opt = optim.RMSprop(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = optim.RMSprop(self.critic2.parameters(), lr=cfg.critic_lr)

        self.buf = ReplayBuffer(cfg.buffer_size)

        self.noise = OUNoise(dim=self.act_dim, 
                             sigma_start=cfg.eps_start, 
                             sigma_decay=cfg.eps_decay, 
                             sigma_min=cfg.eps_end)
        self.eps = float(cfg.eps_start)
        self.total_steps = 0
        self.update_steps = 0

    def reset_episode(self):
        self.noise.reset()

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Returns action vector in env.action_space (clipped to bounds).
        """
        self.actor.eval()
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)

        with torch.no_grad():
            a = self.actor(obs_t).squeeze(0).cpu().numpy().astype(np.float32)

        if explore:
            a = a + self.noise.step(self.rng)

        a = np.clip(a, self.action_low, self.action_high).astype(np.float32)
        return a

    def observe(self, obs, act, rew, next_obs, done) -> None:
        self.buf.add(obs, act, rew, next_obs, done)
        self.total_steps += 1

        # decay epsilon
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def can_update(self) -> bool:
        return (len(self.buf) >= self.cfg.batch_size) and (self.total_steps >= self.cfg.warmup_steps)

    def update(self) -> dict:
        """
        One gradient update step (if enough data).
        """
        if not self.can_update():
            return {}
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        obs, act, rew, next_obs, done = self.buf.sample(self.cfg.batch_size, self.rng)

        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(act, device=self.device)
        rew_t = torch.as_tensor(rew, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        done_t = torch.as_tensor(done, device=self.device)

        gamma = float(self.cfg.gamma)

        with torch.no_grad():
            next_act_t = self.actor_t(next_obs_t)
            q1n = self.critic1_t(next_obs_t, next_act_t)
            q2n = self.critic2_t(next_obs_t, next_act_t)
            qn = torch.minimum(q1n, q2n)
            target = rew_t + gamma * (1.0 - done_t) * qn

        # critics
        q1 = self.critic1(obs_t, act_t)
        q2 = self.critic2(obs_t, act_t)
        loss1 = nn.functional.mse_loss(q1, target)
        loss2 = nn.functional.mse_loss(q2, target)

        self.critic1_opt.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.grad_clip)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.grad_clip)
        self.critic2_opt.step()

        # actor
        self.actor_opt.zero_grad()
        act_pi = self.actor(obs_t)
        actor_loss = -self.critic1(obs_t, act_pi).mean()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        self.update_steps += 1

        if (self.update_steps % int(self.cfg.target_update_every)) == 0:
            self._soft_update(self.actor_t, self.actor, self.cfg.tau)
            self._soft_update(self.critic1_t, self.critic1, self.cfg.tau)
            self._soft_update(self.critic2_t, self.critic2, self.cfg.tau)

        return {
            "loss_actor": float(actor_loss.item()),
            "loss_critic1": float(loss1.item()),
            "loss_critic2": float(loss2.item()),
            "eps": float(self.eps),
        }

    # save/load 
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic1_t": self.critic1_t.state_dict(),
            "critic2_t": self.critic2_t.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "eps": float(self.eps),
            "total_steps": int(self.total_steps),
            "update_steps": int(self.update_steps),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic1.load_state_dict(sd["critic1"])
        self.critic2.load_state_dict(sd["critic2"])
        self.actor_t.load_state_dict(sd["actor_t"])
        self.critic1_t.load_state_dict(sd["critic1_t"])
        self.critic2_t.load_state_dict(sd["critic2_t"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic1_opt.load_state_dict(sd["critic1_opt"])
        self.critic2_opt.load_state_dict(sd["critic2_opt"])
        self.eps = float(sd.get("eps", self.eps))
        self.total_steps = int(sd.get("total_steps", self.total_steps))
        self.update_steps = int(sd.get("update_steps", self.update_steps))

    @staticmethod
    def _hard_update(dst: nn.Module, src: nn.Module) -> None:
        dst.load_state_dict(src.state_dict())

    @staticmethod
    def _soft_update(dst: nn.Module, src: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p in zip(dst.parameters(), src.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)
