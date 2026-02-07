from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import numpy as np


class Policy(Protocol):
    def reset(self, seed: Optional[int] = None) -> None: ...
    def act(self, obs: np.ndarray, t: int = 0) -> Optional[np.ndarray]: ...


@dataclass
class NonePolicy:
    """Policy that returns None (used when watermark doesn't depend on action)."""

    def reset(self, seed: Optional[int] = None) -> None:
        return

    def act(self, obs: np.ndarray, t: int = 0) -> Optional[np.ndarray]:
        return None


def build_policy(policy_cfg: Dict[str, Any], env) -> Policy:
    """Create a policy instance from config.

    Supported types:
      - none
      - zero_action
      - constant_action
      - ddpg
    """
    ptype = str(policy_cfg.get("type", "none")).lower().strip()

    if ptype in ("none", "null"):
        return NonePolicy()

    if ptype in ("zero", "zero_action", "zeros"):
        from src.policies.constant import ZeroActionPolicy
        act_dim = int(getattr(env.action_space, "shape", [0])[0])
        return ZeroActionPolicy(act_dim)

    if ptype in ("constant", "constant_action"):
        from src.policies.constant import ConstantActionPolicy
        U = np.asarray(policy_cfg.get("U"), dtype=float)
        return ConstantActionPolicy(U)

    if ptype in ("ddpg", "rl"):
        return build_ddpg_policy(policy_cfg, env)

    raise ValueError(f"Unknown policy.type='{ptype}'")


def _guess_agent_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    for k in (
        "agent",
        "agent_state",
        "agent_state_dict",
        "ddpg",
        "ddpg_state",
        "state_dict",
        "model",
        "model_state_dict",
    ):
        if k in payload and isinstance(payload[k], dict):
            return payload[k]
    if all(isinstance(v, dict) for v in payload.values()):
        return payload
    raise KeyError("Could not find agent state dict in checkpoint payload")


def build_ddpg_policy(policy_cfg: Dict[str, Any], env):
    """Load a trained DDPG agent and expose it as a deterministic policy."""
    import torch

    from src.agents.ddpg import DDPGAgent, DDPGConfig

    device = str(policy_cfg.get("device", "cpu"))
    explore = bool(policy_cfg.get("explore", False))

    # checkpoint location
    ckpt_path = policy_cfg.get("ckpt_path")
    ckpt_dir = policy_cfg.get("ckpt_dir")
    which = str(policy_cfg.get("which", "best")).lower()

    if ckpt_path is None:
        if ckpt_dir is None:
            raise ValueError("ddpg policy requires ckpt_path or ckpt_dir")
        ckpt_dir = Path(str(ckpt_dir))
        ckpt_path = ckpt_dir / ("ckpt_best.pt" if which == "best" else "ckpt_latest.pt")
    ckpt_path = Path(str(ckpt_path))

    payload = torch.load(ckpt_path, map_location="cpu")

    ddpg_cfg_in = dict(policy_cfg.get("ddpg", {}) or {})
    if not ddpg_cfg_in:
        for k in ("ddpg_cfg", "ddpg", "cfg", "config"):
            v = payload.get(k)
            if isinstance(v, dict) and v:
                ddpg_cfg_in = dict(v)
                break
    if not ddpg_cfg_in:
        snap = ckpt_path.parent / "config_snapshot.json"
        if snap.exists():
            try:
                import json

                snap_cfg = json.loads(snap.read_text())
                if isinstance(snap_cfg, dict) and isinstance(snap_cfg.get("ddpg"), dict):
                    ddpg_cfg_in = dict(snap_cfg["ddpg"])
            except Exception:
                ddpg_cfg_in = {}
    cfg = DDPGConfig(**ddpg_cfg_in)

    obs_dim = int(getattr(env.observation_space, "shape", [0])[0])
    d = int(getattr(env, "d", 1))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    seed = int(policy_cfg.get("seed", 0))

    agent = DDPGAgent(
        obs_dim=obs_dim,
        d=d,
        action_low=action_low,
        action_high=action_high,
        cfg=cfg,
        seed=seed,
        device=device,
    )

    agent_state = _guess_agent_state(payload)
    agent.load_state_dict(agent_state)

    @dataclass
    class _DDPGPolicy:
        agent: DDPGAgent
        explore: bool

        def reset(self, seed: Optional[int] = None) -> None:
            self.agent.reset_episode()

        def act(self, obs: np.ndarray, t: int = 0) -> np.ndarray:
            return self.agent.act(obs, explore=self.explore)

    return _DDPGPolicy(agent=agent, explore=explore)
