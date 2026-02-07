#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
import torch

from env.factory import make_env

from src.agents.ddpg import DDPGAgent, DDPGConfig
from src.train.logger import JsonlLogger, save_json
from src.train.checkpoint import CheckpointManager
from src.plots.curves import write_learning_curve


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _episode_rollout(
    env,
    agent: DDPGAgent,
    *,
    max_steps: int,
    explore: bool,
    seed: Optional[int] = None,
) -> Tuple[float, int, Dict[str, float]]:
    """
    Runs 1 episode and returns (return, length, metrics).
    Metrics are lightweight aggregates.
    """
    obs, info0 = env.reset(seed=seed)
    agent.reset_episode()

    ep_ret = 0.0
    ep_disc_ret = 0.0
    ep_len = 0

    g_sum = 0.0
    it_sum = 0.0
    s1_last = None

    for t in range(int(max_steps)):
        act = agent.act(obs, explore=explore)
        next_obs, rew, terminated, truncated, info = env.step(act)

        done = bool(terminated or truncated)
        if explore:
            agent.observe(obs, act, rew, next_obs, done)

        ep_ret += float(rew)
        ep_disc_ret += float(rew) * (agent.cfg.gamma ** t)
        ep_len += 1

        g_sum += _as_float(info.get("g", info.get("g_end", 0.0)), 0.0)
        it_sum += _as_float(info.get("It", info.get("It_end", 0.0)), 0.0)
        s1_last = info.get("S1", info.get("S1_end", s1_last))

        obs = next_obs

        if explore:
            agent.update()

        if done:
            break

    metrics = {
        "g_mean": float(g_sum / max(ep_len, 1)),
        "It_mean": float(it_sum / max(ep_len, 1)),
        "S1_last": float(s1_last) if s1_last is not None else float("nan"),
    }
    return float(ep_ret), float(ep_disc_ret), int(ep_len), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config for training")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    train_cfg = dict(cfg.get("train", {}))
    out_dir = Path(train_cfg.get("out_dir", "output/train")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "config_snapshot.json", {"config_path": str(cfg_path), "config": cfg})

    env_block = dict(cfg.get("env", {}))
    env_name = str(env_block.get("env_name", "dt")).lower()
    plant_data_path = Path(env_block["plant_data_path"]).expanduser().resolve()
    plant_data = _load_json(plant_data_path)
    env_cfg = dict(env_block.get("env_cfg", {}))

    seed = int(train_cfg.get("seed", 0))
    beta_seed = int(train_cfg.get("beta_seed", env_cfg.get("beta_seed", 0)))

    env = make_env(env_name, plant_data=plant_data, env_cfg=env_cfg, seed=seed, beta_seed=beta_seed)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    d = int(getattr(env, "d", None) or env_cfg.get("d", None) or int(np.round((np.sqrt(8 * act_dim + 1) - 1) / 2)))

    ddpg_block = dict(cfg.get("ddpg", {}))
    ddpg_cfg = DDPGConfig(
        hidden_dim=int(ddpg_block.get("hidden_dim", 128)),
        actor_lr=float(ddpg_block.get("actor_lr", 1e-3)),
        critic_lr=float(ddpg_block.get("critic_lr", 1e-3)),
        gamma=float(ddpg_block.get("gamma", 0.99)),
        tau=float(ddpg_block.get("tau", 5e-3)),
        buffer_size=int(ddpg_block.get("buffer_size", 1_000_000)),
        batch_size=int(ddpg_block.get("batch_size", 256)),
        warmup_steps=int(ddpg_block.get("warmup_steps", 5_000)),
        eps_start=float(ddpg_block.get("eps_start", 0.2)),
        eps_end=float(ddpg_block.get("eps_end", 0.02)),
        eps_decay=float(ddpg_block.get("eps_decay", 0.9995)),
        target_update_every=int(ddpg_block.get("target_update_every", 1)),
        grad_clip=float(ddpg_block.get("grad_clip", 1.0)),
    )

    agent = DDPGAgent(
        obs_dim=obs_dim,
        d=d,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        cfg=ddpg_cfg,
        seed=seed,
    )

    ep_logger = JsonlLogger(out_dir / "episodes.jsonl")
    eval_logger = JsonlLogger(out_dir / "eval.jsonl")
    ckpt = CheckpointManager(out_dir)

    resume = bool(train_cfg.get("resume", True))
    start_ep = 0
    best_eval = -float("inf")

    if resume and ckpt.latest_path.exists():
        payload = ckpt.load("latest", map_location="cpu")
        agent.load_state_dict(payload["agent"])
        start_ep = int(payload.get("episode", 0)) + 1
        best_eval = float(payload.get("best_eval", -float("inf")))
        if "np_rng_state" in payload:
            agent.rng.bit_generator.state = payload["np_rng_state"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        print(f"[resume] loaded {ckpt.latest_path} (start_ep={start_ep}, best_eval={best_eval:.4f})")

    episodes = int(train_cfg.get("episodes", 200))
    max_steps = int(train_cfg.get("max_steps", 10_000)) 
    eval_every = int(train_cfg.get("eval_every", 10))
    eval_episodes_n = int(train_cfg.get("eval_episodes", 3))
    ckpt_every = int(train_cfg.get("ckpt_every", 5))

    curve_ep = []
    curve_train_ret = []
    curve_train_len = []
    curve_eval_ep = []
    curve_eval_ret = []

    for ep in range(start_ep, episodes):
        ep_seed = seed + ep
        ret, disc_ret, length, metrics = _episode_rollout(env, agent, max_steps=max_steps, explore=True, seed=ep_seed)

        row = {
            "episode": ep,
            "seed": ep_seed,
            "return": ret,
            "discounted_return": disc_ret,
            "length": length,
            "eps": float(getattr(agent, "eps", float("nan"))),
            "total_steps": int(getattr(agent, "total_steps", 0)),
            **metrics,
        }
        ep_logger.write(row)

        curve_ep.append(ep)
        curve_train_ret.append(ret)
        curve_train_len.append(length)

        did_eval = False
        eval_mean = None
        if (ep + 1) % eval_every == 0:
            did_eval = True
            eval_rets = []
            eval_disc_rets = []
            for j in range(eval_episodes_n):
                rj, disc_rj, lj, mj = _episode_rollout(
                    env, agent, max_steps=max_steps, explore=False, seed=seed + 10_000 + ep * 100 + j
                )
                eval_rets.append(rj)
                eval_disc_rets.append(disc_rj)

            eval_mean = float(np.mean(eval_rets)) if eval_rets else float("nan")
            eval_disc_mean = float(np.mean(eval_disc_rets)) if eval_disc_rets else float("nan")
            eval_logger.write({"episode": ep, "eval_mean_return": eval_mean, "eval_mean_discounted_return": eval_disc_mean, "eval_returns": eval_rets})

            curve_eval_ep.append(ep)
            curve_eval_ret.append(eval_mean)

        if (ep + 1) % ckpt_every == 0 or ep == episodes - 1:
            payload = {
                "episode": ep,
                "best_eval": float(best_eval),
                "agent": agent.state_dict(),
                "ddpg_cfg": asdict(ddpg_cfg),
                "env_name": env_name,
                "np_rng_state": agent.rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
            }
            ckpt.save_latest(payload)

        if did_eval and eval_mean is not None and np.isfinite(eval_mean) and eval_mean > best_eval:
            best_eval = float(eval_mean)
            payload = {
                "episode": ep,
                "best_eval": float(best_eval),
                "agent": agent.state_dict(),
                "ddpg_cfg": asdict(ddpg_cfg),
                "env_name": env_name,
                "np_rng_state": agent.rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
            }
            ckpt.save_best(payload)
            print(f"[best] episode={ep} eval_mean_return={best_eval:.4f}")

        save_json(
            out_dir / "learning_curve.json",
            {
                "episodes": curve_ep,
                "train_return": curve_train_ret,
                "train_length": curve_train_len,
                "eval_episodes": curve_eval_ep,
                "eval_return": curve_eval_ret,
            },
        )
        write_learning_curve(
            out_png=out_dir / "learning_curve.png",
            episodes=curve_ep,
            train_return=curve_train_ret,
            train_len=curve_train_len,
            eval_return=curve_eval_ret,
            eval_episodes=curve_eval_ep,
        )

        if (ep + 1) % 5 == 0:
            print(f"[train] ep={ep + 1} return={ret:.3f} discounted_return={disc_ret:.3f}")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()


# python -m src.train.train_ddpg --config configs/train_ddpg.yaml