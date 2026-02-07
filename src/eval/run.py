from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional

import yaml

from .history import NominalHistory, collect_nominal_history
from .policies import build_policy
from .rollout import run_one_rep
from .utils import find_repo_root, jsonify, tag_attack, tag_watermark


def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def resolve_out_base(env_name: str, eval_cfg: Dict[str, Any]) -> Path:
    if "out_dir" in eval_cfg and eval_cfg["out_dir"]:
        return Path(str(eval_cfg["out_dir"])).expanduser().resolve()
    out_root = Path(str(eval_cfg.get("out_root", "output"))).expanduser().resolve()
    return out_root / env_name / "simulation"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_cfg(cfg_path)

    repo_root = find_repo_root(Path(__file__).resolve().parent)
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    env_block = dict(cfg.get("env", {}))
    env_name = str(env_block.get("env_name"))
    if not env_name:
        raise ValueError("env.env_name is required")

    plant_data_path = Path(str(env_block.get("plant_data_path"))).expanduser().resolve()
    if not plant_data_path.exists():
        plant_data_path = (repo_root / str(env_block.get("plant_data_path"))).resolve()
    plant_data = load_json(plant_data_path)

    env_cfg = dict(env_block.get("env_cfg", {}))

    eval_cfg = dict(cfg.get("eval", cfg.get("sim", {})))
    out_base = resolve_out_base(env_name, eval_cfg)

    overwrite = bool(eval_cfg.get("overwrite", False))
    steps_env = int(eval_cfg.get("steps_env", eval_cfg.get("max_steps", 2000)))
    reps = int(eval_cfg.get("reps", 1))
    base_seed = int(eval_cfg.get("base_seed", eval_cfg.get("seed", 0)))
    base_beta_seed = int(eval_cfg.get("base_beta_seed", eval_cfg.get("beta_seed", 0)))

    default_policy_cfg = dict(cfg.get("policy", {"type": "none"}))
    default_wm_cfg = dict(cfg.get("watermark", {"mode": "none"}))
    default_attack_cfg = dict(cfg.get("attack", {"type": "none"}))

    runs = cfg.get("runs", None)
    if not runs:
        runs = [
            {
                "policy": default_policy_cfg,
                "watermark": default_wm_cfg,
                "attack": default_attack_cfg,
            }
        ]

    for run in runs:
        policy_cfg = {**default_policy_cfg, **dict(run.get("policy", {}))}
        wm_cfg = {**default_wm_cfg, **dict(run.get("watermark", {}))}
        attack_cfg = {**default_attack_cfg, **dict(run.get("attack", {}))}

        wm_tag = tag_watermark(wm_cfg)
        atk_tag = tag_attack(attack_cfg)

        run_name = run.get("name", None)
        use_name = bool(eval_cfg.get("use_run_name_subdir", False))
        run_root = out_base / (str(run_name) if (use_name and run_name) else "") / wm_tag / atk_tag
        run_root = run_root.resolve()

        if overwrite and run_root.exists():
            shutil.rmtree(run_root)

        from env.factory import make_env

        tmp_env = make_env(env_name, plant_data=plant_data, env_cfg=env_cfg, seed=base_seed, beta_seed=base_beta_seed)
        policy = build_policy(policy_cfg, tmp_env)

        nominal_hist: Optional[NominalHistory] = None
        atype = str(attack_cfg.get("type", "none")).lower().strip()
        if atype == "replay":
            hist_cfg = dict(attack_cfg.get("replay", {}).get("history", {"mode": "run_nominal_once"}))
            mode = str(hist_cfg.get("mode", "run_nominal_once")).lower()

            if mode == "run_nominal_once":
                hist_seed = int(hist_cfg.get("seed", base_seed))
                hist_beta_seed = int(hist_cfg.get("beta_seed", base_beta_seed))
                hist_steps = int(hist_cfg.get("steps_env", steps_env))

                nominal_hist = collect_nominal_history(
                    env_name=env_name,
                    plant_data=plant_data,
                    env_cfg=env_cfg,
                    steps_env=hist_steps,
                    seed=hist_seed,
                    beta_seed=hist_beta_seed,
                    policy=policy,
                    wm_cfg=wm_cfg,
                )

                hist_dir = run_root / "_replay_history"
                hist_dir.mkdir(parents=True, exist_ok=True)
                with open(hist_dir / "nominal_history.json", "w") as f:
                    json.dump(
                        jsonify(
                            {
                                "y_dec": nominal_hist.y_dec,
                                "y_fast_blocks": nominal_hist.y_fast_blocks,
                                "n_fast": nominal_hist.n_fast,
                            }
                        ),
                        f,
                        indent=2,
                    )
            else:
                raise ValueError(f"Unsupported attack.replay.history.mode='{mode}'")

        # run replications
        summaries: List[Dict[str, Any]] = []
        for rep in tqdm(range(reps), 
                        desc=f"Simulating: env='{env_name}' | wm='{wm_tag}' | atk='{atk_tag}'", 
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        unit="rep"):
            rep_seed = base_seed + rep
            rep_beta_seed = base_beta_seed + rep
            rep_dir = run_root / f"rep{rep:02d}"

            summary = run_one_rep(
                out_dir=rep_dir,
                env_name=env_name,
                plant_data=plant_data,
                env_cfg=env_cfg,
                steps_env=steps_env,
                rep=rep,
                seed=rep_seed,
                beta_seed=rep_beta_seed,
                policy=policy,
                wm_cfg=wm_cfg,
                attack_cfg=attack_cfg,
                nominal_hist=nominal_hist,
            )
            summaries.append(summary)

        run_root.mkdir(parents=True, exist_ok=True)
        with open(run_root / "run_summary.json", "w") as f:
            json.dump(jsonify({"summaries": summaries, "config": cfg}), f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
