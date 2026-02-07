from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .step import advance_one
from .utils import as_1d


@dataclass
class NominalHistory:
    """History required for replay attacks."""
    y_dec: List[List[float]]
    y_fast_blocks: Optional[List[List[float]]] = None
    n_fast: Optional[int] = None


def build_replay_override(
    env_name: str,
    *,
    k: int,
    onset_t: int,
    hist: NominalHistory,
    replay_cfg: Dict[str, Any],
):
    """Return next_y_override for env.plant.step(next_y_override=...).
    """
    if k < onset_t:
        return None

    start_idx = int(replay_cfg.get("start_idx", 0))
    wrap = bool(replay_cfg.get("wrap", False))

    def _get_idx(L: int, rel: int) -> Optional[int]:
        if L <= 0:
            return None
        idx = start_idx + rel
        if wrap:
            idx = idx % L
        else:
            idx = min(idx, L - 1)
        return idx

    rel = int(k - onset_t)

    if str(env_name).lower().strip() in ("sm_dt_disc", "sm_dt", "sm_dt_discrete", "sm-dt-discrete"):
        if hist.y_fast_blocks is not None:
            idx = _get_idx(len(hist.y_fast_blocks), rel)
            if idx is None:
                return None
            return np.asarray(hist.y_fast_blocks[idx], dtype=float).reshape(-1)

    idx = _get_idx(len(hist.y_dec), rel)
    if idx is None:
        return None
    return np.asarray(hist.y_dec[idx], dtype=float)


def collect_nominal_history(
    *,
    env_name: str,
    plant_data: Dict[str, Any],
    env_cfg: Dict[str, Any],
    steps_env: int,
    seed: int,
    beta_seed: int,
    policy,
    wm_cfg: Dict[str, Any],
):
    """Run ONE nominal rollout to capture nominal outputs for replay attacks."""
    from env.factory import make_env
    from .watermark import select_U

    env = make_env(env_name, plant_data=plant_data, env_cfg=env_cfg, seed=seed, beta_seed=beta_seed)
    obs, _ = env.reset(seed=seed)

    rng = np.random.default_rng(seed)

    y_dec: List[List[float]] = []
    y_fast_blocks: List[List[float]] = []
    n_fast: Optional[int] = None
    saw_fast = False

    for k in range(int(steps_env)):
        action = policy.act(obs, t=k) if policy is not None else None
        U = select_U(wm_cfg, env, action, rng)
        obs, _, terminated, _, ctx, _ = advance_one(env, U=U, next_y_override=None)

        # decision-time output
        y_dec.append(as_1d(env.plant.y_curr).tolist())

        if "y_fast" in ctx:
            saw_fast = True
            y_fast = np.asarray(ctx["y_fast"], dtype=float).reshape(-1)
            y_fast_blocks.append(y_fast.tolist())
            n_fast = int(len(y_fast))

        if terminated:
            break

    return NominalHistory(y_dec=y_dec, y_fast_blocks=y_fast_blocks if saw_fast else None, n_fast=n_fast)
