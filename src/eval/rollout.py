from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .history import NominalHistory, build_replay_override
from .io import write_jsonl
from .step import advance_one
from .utils import as_1d, jsonify, write_json
from .watermark import select_U


def run_one_rep(
    *,
    out_dir: Path,
    env_name: str,
    plant_data: Dict[str, Any],
    env_cfg: Dict[str, Any],
    steps_env: int,
    rep: int,
    seed: int,
    beta_seed: int,
    policy,
    wm_cfg: Dict[str, Any],
    attack_cfg: Dict[str, Any],
    nominal_hist: Optional[NominalHistory],
) -> Dict[str, Any]:
    """Run a single replication, writing per-rep files."""
    from env.factory import make_env  # noqa: WPS433

    env = make_env(env_name, plant_data=plant_data, env_cfg=env_cfg, seed=seed, beta_seed=beta_seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    atype = str(attack_cfg.get("type", "none")).lower().strip()
    onset_t = int(attack_cfg.get("onset_t", 10**18))

    decision_rows: List[Dict[str, Any]] = []
    processed_rows: List[Dict[str, Any]] = []

    if hasattr(policy, "reset"):
        policy.reset(seed=seed)

    for k in range(int(steps_env)):
        # choose watermark
        action = policy.act(obs, t=k) if policy is not None else None
        U = select_U(wm_cfg, env, action, rng)

        # choose attack override
        next_y_override = None
        active_attack = False
        if atype == "replay":
            if nominal_hist is None:
                raise ValueError("Replay attack requires nominal history; set attack.replay.history or run a normal history.")
            next_y_override = build_replay_override(
                env_name,
                k=k,
                onset_t=onset_t,
                hist=nominal_hist,
                replay_cfg=dict(attack_cfg.get("replay", {})),
            )
            active_attack = (k >= onset_t) and (next_y_override is not None)

        # advance
        obs_next, _, terminated, info, ctx, processed = advance_one(env, U=U, next_y_override=next_y_override)

        # store decision row (1 row / decision epoch)
        row = {
            "rep": int(rep),
            "k": int(k),
            "t_dec": int(getattr(env, "t_dec", k + 1)),
            "attack_active": bool(active_attack),
            "attack_type": atype,
            "onset_t": int(onset_t),
            "y": as_1d(env.plant.y_curr).tolist(),
            "S1": float(getattr(env.belief, "S1", 0.0)),
            "U": np.asarray(U, dtype=float).reshape(-1).tolist(),
            "action": None if action is None else np.asarray(action, dtype=float).reshape(-1).tolist(),
            "terminated": bool(terminated),
            **jsonify(info),
        }
        if "U_applied" in ctx:
            row["U_applied"] = np.asarray(ctx["U_applied"], dtype=float).reshape(-1).tolist()
        decision_rows.append(row)

        # store processed rows for batched plant
        if processed:
            t_fast_start = ctx.get("t_fast_start", None)
            for pr in processed:
                pr_row = {
                    "rep": int(rep),
                    "k": int(k),
                    "t_dec": int(getattr(env, "t_dec", k + 1)),
                    "t_fast_start": None if t_fast_start is None else int(t_fast_start),
                    "t_fast": None if t_fast_start is None else int(t_fast_start) + int(pr["i"]),
                    "U": np.asarray(U, dtype=float).reshape(-1).tolist(),
                    "action": None if action is None else np.asarray(action, dtype=float).reshape(-1).tolist(),
                    "attack_active": bool(active_attack),
                    "attack_type": atype,
                    "onset_t": int(onset_t),
                    **jsonify(pr),
                }
                L = len(processed)
                i = int(pr["i"])
                pr_row["t_proc"] = int(k) * int(L) + i
                pr_row["proc_len"] = int(L)
                if "U_applied" in ctx:
                    pr_row["U_applied"] = np.asarray(ctx["U_applied"], dtype=float).reshape(-1).tolist()
                processed_rows.append(pr_row)

        obs = obs_next
        if terminated:
            break

    out_dir.mkdir(parents=True, exist_ok=True)

    # write files
    write_jsonl(out_dir / "traj_decision.jsonl", decision_rows)
    if processed_rows:
        write_jsonl(out_dir / "traj_processed.jsonl", processed_rows)

    summary = {
        "rep": int(rep),
        "steps_executed": int(len(decision_rows)),
        "return": float(sum(float(r.get("reward_total", r.get("reward", 0.0))) for r in decision_rows)),
        "terminated": bool(decision_rows[-1]["terminated"]) if decision_rows else False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary
