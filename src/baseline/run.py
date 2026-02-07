from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .io import write_json, ensure_dir
from .rollout import learn_tac_once, collect_nominal_y_history, simulate_rep


def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def find_repo_root(start: Path) -> Path:
    """Walk up to find repo root (contains 'env' and 'src')."""
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "env").exists() and (p / "src").exists():
            return p
    return cur.parents[len(cur.parents) - 1]


def resolve_out_base(env_name: str, eval_cfg: Dict[str, Any]) -> Path:
    if eval_cfg.get("out_dir"):
        return Path(str(eval_cfg["out_dir"])).expanduser().resolve()
    out_root = Path(str(eval_cfg.get("out_root", "output"))).expanduser().resolve()
    return out_root / env_name / "simulation"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_learning_plots(learn_dir: Path) -> None:
    """Save y, g_hat, U curves into learn_dir, based on learn_traj.jsonl."""
    traj_path = learn_dir / "learn_traj.jsonl"
    if not traj_path.exists():
        print(f"[WARN] No learn_traj.jsonl at {traj_path}, skipping plots.")
        return

    rows = read_jsonl(traj_path)
    if not rows:
        print(f"[WARN] learn_traj.jsonl empty, skipping plots.")
        return

    t = np.array([r["t"] for r in rows], dtype=int)
    y = np.array([r["y"] for r in rows], dtype=float)

    g = np.array([np.nan if r.get("g_hat") is None else float(r["g_hat"]) for r in rows], dtype=float)

    if "U_next" in rows[0]:
        U = np.array([r["U_next"] for r in rows], dtype=float)
    else:
        U = np.array([r["U_internal"] for r in rows], dtype=float)

    ensure_dir(learn_dir)

    # y plot
    plt.figure()
    plt.plot(t, y, label=r"plant output $y_{t+1}$")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(learn_dir / "learn_y.png", dpi=200)
    plt.close()

    # g plot (log scale, only if there are positive finite values)
    plt.figure()
    plt.plot(t, g, label=r"$\hat{g}_t$")
    plt.xlabel("t")
    plt.ylabel(r"detection statistic $\hat{g}_t$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Use log scale only if safe
    if np.isfinite(g).any() and np.nanmax(g) > 0:
        plt.yscale("log")
    plt.savefig(learn_dir / "learn_g.png", dpi=200)
    plt.close()

    # U plot
    plt.figure()
    plt.plot(t, U, label=r"$U$ (watermark covariance)")
    plt.xlabel("t")
    plt.ylabel("U")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(learn_dir / "learn_U.png", dpi=200)
    plt.close()


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

    env_cfg = dict(cfg.get("env", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    tac_cfg = dict(cfg.get("tac", {}))
    detector_cfg = dict(cfg.get("detector", {}))
    atk_cfg = dict(cfg.get("attack", {}))
    plot_cfg = dict(cfg.get("plots", {}))

    env_name = str(env_cfg.get("env_name", "sm"))
    plant_data_path = Path(str(env_cfg.get("plant_data_path"))).expanduser().resolve()
    if not plant_data_path.exists():
        plant_data_path = (repo_root / str(env_cfg.get("plant_data_path"))).resolve()
    if not plant_data_path.exists():
        raise FileNotFoundError(f"plant_data_path not found: {plant_data_path}")

    plant_data = load_json(plant_data_path)

    out_base = resolve_out_base(env_name, eval_cfg)
    ensure_dir(out_base)

    overwrite = bool(eval_cfg.get("overwrite", False))
    steps_env = int(eval_cfg.get("steps_env", 2000))
    reps = int(eval_cfg.get("reps", 1))
    base_seed = int(eval_cfg.get("base_seed", 0))

    # deltas
    deltas = tac_cfg.get("deltas", None)
    if deltas is None:
        raise ValueError("tac.deltas is required in YAML (list of floats)")
    deltas = [float(x) for x in deltas]
    if not deltas:
        raise ValueError("tac.deltas is empty")

    # learning phase
    learn_seed = int(tac_cfg.get("learn_seed", 123))
    learn_steps = int(tac_cfg.get("learn_steps", 3000))
    n_eigs = int(tac_cfg.get("n_eigs", 2))
    beta = float(tac_cfg.get("beta", 1.0 / 3.0))
    update_interval = int(tac_cfg.get("update_interval", 100))
    min_k_for_updates = int(tac_cfg.get("min_k_for_updates", 200))

    # detector settings
    false_alarm_rate = float(detector_cfg.get("false_alarm_rate", 0.01))
    eta_burn_in = int(detector_cfg.get("eta_burn_in", 5000))

    # replay settings
    attack_onset = int(atk_cfg.get("attack_onset", 800))
    attack_len = int(atk_cfg.get("attack_len", 600))
    replay_from = int(atk_cfg.get("replay_from", 0))
    history_seed = int(atk_cfg.get("history_seed", 999))

    # plotting
    save_plots = bool(plot_cfg.get("save_learning_plots", True))

    for delta in deltas:
        wm_tag = f"delta={delta:g}"
        run_root = (out_base / wm_tag).resolve()

        if overwrite and run_root.exists():
            shutil.rmtree(run_root)

        ensure_dir(run_root)
        write_json(
            run_root / "config.json",
            {
                "config_path": str(cfg_path),
                "env": env_cfg,
                "eval": eval_cfg,
                "tac": tac_cfg,
                "attack": atk_cfg,
                "plots": plot_cfg,
                "delta": float(delta),
            },
        )

        learn_dir = run_root / "learn" 
        final_state = learn_tac_once(
            out_dir=learn_dir,
            plant_data=plant_data,
            seed=learn_seed,
            steps=learn_steps,
            n_eigs=n_eigs,
            delta=float(delta),
            beta=beta,
            update_interval=update_interval,
            min_k_for_updates=min_k_for_updates,
            false_alarm_rate=false_alarm_rate,
            eta_burn_in=eta_burn_in,
        )

        if save_plots:
            save_learning_plots(learn_dir)

        U_star = float(final_state["U_star"])
        if not (U_star > 0.0):
            raise RuntimeError(f"Learned U_star is not positive for delta={delta}: U_star={U_star}")
        
        eta = final_state.get("eta", None)
        print(f"Learned U_star={U_star:g} for delta={delta:g} with eta={eta}")

        hist_dir = run_root / "_replay_history"
        y_hist = collect_nominal_y_history(
            out_dir=hist_dir,
            plant_data=plant_data,
            seed=history_seed,
            steps=steps_env,
            U_star_internal=U_star,
        )

        if attack_len < 0:
            max_replay_len = len(y_hist) - replay_from
            if max_replay_len <= 0:
                raise ValueError(f"Replay history too short for replay_from={replay_from}: len={len(y_hist)}")
            attack_len = min(steps_env - attack_onset, max_replay_len)

        need = replay_from + attack_len
        if len(y_hist) < need:
            raise ValueError(
                f"Replay history too short: len={len(y_hist)} need>={need}. "
                f"Either increase history length, reduce replay_from/attack_len, or set attack_len:-1."
            )

        summaries: List[Dict[str, Any]] = []

        for scenario in ["normal", "attack"]:
            is_attack = (scenario == "attack")
            scen_root = run_root / scenario
            ensure_dir(scen_root)

            for rep in tqdm(range(reps), desc=f"delta={delta:g} | {scenario}", unit="rep"):
                rep_seed = base_seed + rep
                rep_dir = scen_root / f"rep{rep:02d}"

                summary = simulate_rep(
                    out_dir=rep_dir,
                    plant_data=plant_data,
                    seed=rep_seed,
                    rep=rep,
                    eta=eta,
                    steps=steps_env,
                    U_star_internal=U_star,
                    learned_W_hat=final_state.get("W_hat", None),
                    learned_Uphi_hat=final_state.get("Uphi_hat", None),
                    learned_roots=final_state.get("roots", None),
                    learned_Omega=final_state.get("Omega", None),
                    n_eigs=n_eigs,
                    attack=is_attack,
                    y_replay_hist=y_hist if is_attack else None,
                    attack_onset=attack_onset,
                    attack_len=attack_len,
                    replay_from=replay_from,
                )
                summary["delta"] = float(delta)
                summary["scenario"] = scenario
                summaries.append(summary)

        write_json(run_root / "run_summary.json", {"delta": float(delta), "summaries": summaries})

    print("Done.")


if __name__ == "__main__":
    main()















# """
# Example: run TAC online watermarking on the stepper-motor digital-twin plant.

# Key integration points with existing SMDTContinuousPlant:
#   - plant.step(U, next_y_override=...) returns StepOut with fields:
#       y_curr (y_{t+1}), u (u_cmd = u_nom + phi), phi (phi_t)
#   - Recover u_nom for TAC Section III-B as: u_nom = u_cmd - phi
#   - If your plant rescales U internally (your sm_dt_continuous.py multiplies U),
#     then *also* rescale before giving U_k to TAC (or modify the plant to log U_used).
# """

# import numpy as np

# from env.plants.sm_dt_continuous import SMDTContinuousPlant
# from .online import TACOnlineWatermarker


# def run_rollout(plant, *, n_eigs: int = 2):
#     # Include control in augmented output: y_dim=1, u_dim=1 => m=2
#     tac = TACOnlineWatermarker(
#         y_dim=1,
#         include_u_dim=1,
#         phi_dim=1,
#         n_eigs=n_eigs,
#         delta=0.1,            # <-- tune
#         beta=1/3,
#         update_interval=100,
#         min_k_for_updates=200,
#         Xyy=np.eye(2),
#         Xphiphi=np.eye(1),
#     )
#     plant.reset(seed=42)

#     # Pre-generate nominal y history if you want replay to use actual stored outputs.
#     y_hist = []

#     g_hist = []
#     alarm_hist = []
#     U_hist = []
#     done = False
#     while not done:
#         U_env = tac.current_U()  # covariance for sampling phi_t

#         # Step plant (no replay override for nominal)
#         out = plant.step(U_env)  # StepOut
#         y_next = out.y_curr.reshape(-1)   # (1,)
#         u_cmd = out.u.reshape(-1)         # (1,) includes watermark
#         phi = out.phi.reshape(-1)         # (1,)

#         # controller command before watermark:
#         u_nom = u_cmd - phi

#         # Apply replay by overriding next output in subsequent step, if you implement that mode.
#         # (Your plant currently flips control when override is used; you may want to disable that
#         #  for TAC baseline to match the paper's replay model.)
#         y_hist.append(float(y_next[0]))
#         done = out.terminated

#         # Update TAC with (y_{t+1}, u_t, phi_t)
#         info = tac.step(y_next=y_next, u_ctrl_k=u_nom, phi_k=phi, U_k=U_env)

#         g = info["g_hat"]
#         g_hist.append(g)
#         U_hist.append(info["U_next"][0, 0])

#     return {
#         "g": np.array(g_hist),
#         "U": np.array(U_hist),
#         "y": np.array(y_hist),
#     }


# if __name__ == "__main__":
#     import argparse
#     import json
#     import matplotlib.pyplot as plt


#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data-dir", type=str, default=None, help="Path to sm data.json")
#     args = ap.parse_args()
#     data_dir = args.data_dir 
#     with open(data_dir, "r") as f:
#         data = json.load(f)
#     plant = SMDTContinuousPlant(data=data, seed=42)

#     results = run_rollout(
#         plant,
#         n_eigs=2,
#     )

#     g = results["g"]
#     U = results["U"]
#     y = results["y"]

#     plt.figure()
#     plt.plot(y, label=r"plant output $y_t$")
#     plt.xlabel("Time")
#     plt.ylabel(r"Output $y_t$")
#     plt.legend()
#     plt.grid()

#     plt.figure()
#     plt.plot(g, label=r"$\hat{g}_t$")
#     plt.yscale("log")
#     plt.xlabel("Time")
#     plt.ylabel(r"detection statistic $\hat{g}_t$")
#     plt.legend()
#     plt.grid()

#     plt.figure()
#     plt.plot(U, label=r"$U$ (watermark covariance)")
#     plt.xlabel("Time")
#     plt.ylabel(r"Watermark Covariance $U$")
#     plt.legend()
#     plt.grid()

#     plt.show()
