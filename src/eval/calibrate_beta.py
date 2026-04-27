#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
import numpy as np
import yaml

from env.plants.msd_nonlinear import MSDNonlinearPlant
from env.core.detector import ChiSquareDetector


@dataclass
class CalibrationConfig:
    plant_data_path: Path
    out_dir: Path
    alpha: float
    nominal_rollouts: int
    attack_rollouts: int
    base_seed: int
    history_seed_offset: int
    attack_seed_offset: int
    U_grid: List[float]
    tau_grid: List[int]
    tau_weighting: str
    tau_weights: Optional[List[float]]
    replay_start_mode: str
    replay_start_idx: Optional[int]
    wrap_history: bool
    noise_override: Optional[Dict[str, Any]]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_cfg(cfg_path: Path) -> CalibrationConfig:
    cfg = _load_yaml(cfg_path)
    env_block = dict(cfg.get("env", {}))
    cal = dict(cfg.get("calibration", {}))
    det = dict(env_block.get("detector", cfg.get("detector", {})))

    plant_data_path = Path(env_block["plant_data_path"]).expanduser().resolve()
    out_dir = Path(cal.get("out_dir", "output/msd/calibration")).expanduser().resolve()

    tau_weights = cal.get("tau_weights", None)
    if tau_weights is not None:
        tau_weights = [float(x) for x in tau_weights]

    return CalibrationConfig(
        plant_data_path=plant_data_path,
        out_dir=out_dir,
        alpha=float(det.get("alpha", cal.get("alpha", 0.005))),
        nominal_rollouts=int(cal.get("nominal_rollouts", 200)),
        attack_rollouts=int(cal.get("attack_rollouts", 200)),
        base_seed=int(cal.get("base_seed", 123)),
        history_seed_offset=int(cal.get("history_seed_offset", 10_000)),
        attack_seed_offset=int(cal.get("attack_seed_offset", 20_000)),
        U_grid=[float(x) for x in cal["U_grid"]],
        tau_grid=[int(x) for x in cal["tau_grid"]],
        tau_weighting=str(cal.get("tau_weighting", "uniform")).lower(),
        tau_weights=tau_weights,
        replay_start_mode=str(cal.get("replay_start_mode", "match_onset")).lower(),
        replay_start_idx=(None if cal.get("replay_start_idx", None) is None else int(cal.get("replay_start_idx"))),
        wrap_history=bool(cal.get("wrap_history", False)),
        noise_override=cal.get("noise_override", None),
    )


def _u_to_matrix(U_scalar: float) -> np.ndarray:
    return np.array([[float(U_scalar)]], dtype=float)


def _normalize_weights(tau_grid: Sequence[int], weighting: str, tau_weights: Optional[Sequence[float]], horizon: int) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=int)
    if tau_weights is not None:
        w = np.asarray(tau_weights, dtype=float)
        if w.shape[0] != tau_grid.shape[0]:
            raise ValueError("tau_weights must have the same length as tau_grid")
        s = float(np.sum(w))
        if s <= 0:
            raise ValueError("tau_weights must sum to a positive value")
        return w / s

    if weighting == "uniform":
        return np.full(tau_grid.shape[0], 1.0 / max(tau_grid.shape[0], 1), dtype=float)

    if weighting == "geometric":
        p = 1.0 / float(horizon)
        # Geometric on support {1,2,...}; then renormalize over the chosen tau grid.
        w = p * np.power(1.0 - p, tau_grid - 1)
        return w / np.sum(w)

    raise ValueError("tau_weighting must be one of {'uniform','geometric'} unless tau_weights is supplied")


def _make_plant(plant_data: Dict[str, Any], seed: int) -> MSDNonlinearPlant:
    return MSDNonlinearPlant(plant_data, seed=seed)


def _run_nominal_rollout(plant_data: Dict[str, Any], *, seed: int, U_scalar: float, detector: Optional[ChiSquareDetector] = None) -> Dict[str, Any]:
    plant = _make_plant(plant_data, seed=seed)
    T = int(plant.T)
    U = _u_to_matrix(U_scalar)

    y_hist = []
    g_hist = []
    I_hist = []
    r_hist = []

    for _ in range(T):
        out = plant.step(U)
        y_hist.append(np.asarray(out.y_curr, dtype=float).reshape(-1).copy())
        r_hist.append(np.asarray(out.r, dtype=float).reshape(-1).copy())
        if detector is not None:
            It, g = detector.test(out.r, out.Q)
            I_hist.append(int(It))
            g_hist.append(float(g))

    result = {
        "y_hist": np.asarray(y_hist, dtype=float),
        "r_hist": np.asarray(r_hist, dtype=float),
    }
    if detector is not None:
        result["g_hist"] = np.asarray(g_hist, dtype=float)
        result["I_hist"] = np.asarray(I_hist, dtype=int)
    return result


def _replay_value(history_y: np.ndarray, step_idx: int, tau: int, replay_start_idx: int, wrap_history: bool) -> Optional[np.ndarray]:
    if step_idx < tau:
        return None
    j = replay_start_idx + (step_idx - tau)
    if wrap_history:
        j = j % history_y.shape[0]
    if j < 0 or j >= history_y.shape[0]:
        return None
    return history_y[j].reshape(-1, 1)


def _run_replay_attack_rollout(
    plant_data: Dict[str, Any],
    *,
    seed_attack: int,
    history_y: np.ndarray,
    U_scalar: float,
    tau: int,
    replay_start_idx: int,
    wrap_history: bool,
    detector: ChiSquareDetector,
) -> Dict[str, Any]:
    plant = _make_plant(plant_data, seed=seed_attack)
    T = int(plant.T)
    U = _u_to_matrix(U_scalar)

    g_hist = np.zeros(T, dtype=float)
    I_hist = np.zeros(T, dtype=int)

    for t in range(T):
        next_y_override = _replay_value(history_y, t, tau, replay_start_idx, wrap_history)
        out = plant.step(U, next_y_override=next_y_override)
        It, g = detector.test(out.r, out.Q)
        I_hist[t] = int(It)
        g_hist[t] = float(g)

    return {"g_hist": g_hist, "I_hist": I_hist}


def _choose_threshold_from_nominal(g_nom_all: np.ndarray, alpha: float) -> float:
    return float(np.quantile(g_nom_all, 1.0 - alpha))


def _resolve_replay_start_idx(mode: str, tau: int, replay_start_idx: Optional[int]) -> int:
    if mode == "match_onset":
        return int(tau)
    if mode == "fixed":
        if replay_start_idx is None:
            raise ValueError("replay_start_idx must be provided when replay_start_mode='fixed'")
        return int(replay_start_idx)
    raise ValueError("replay_start_mode must be one of {'match_onset','fixed'}")


def calibrate_lookup(cfg: CalibrationConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plant_data = _load_json(cfg.plant_data_path)
    if cfg.noise_override:
        plant_data = _deep_update(plant_data, cfg.noise_override)

    T = int(plant_data["T"])
    detector_for_nominal = ChiSquareDetector(alpha=cfg.alpha, dof=2)

    # 1) Nominal calibration for a single global threshold g_tilde.
    g_nom_runs = []
    for m in tqdm(range(cfg.nominal_rollouts), desc="Nominal Rollouts for g_tilde Calibration ... ", unit="rollout"):
        seed = cfg.base_seed + m
        nominal = _run_nominal_rollout(
            plant_data,
            seed=seed,
            U_scalar=0.0,
            detector=detector_for_nominal,
        )
        g_nom_runs.append(nominal["g_hist"])
    g_nom_all = np.concatenate(g_nom_runs, axis=0)
    g_tilde = _choose_threshold_from_nominal(g_nom_all, cfg.alpha)

    # Detector with empirical threshold.
    detector = ChiSquareDetector(alpha=cfg.alpha, dof=2)
    detector.gamma = float(g_tilde)
    print(f"Calibrated detector threshold g_tilde: {g_tilde:.4f}")

    tau_weights = _normalize_weights(cfg.tau_grid, cfg.tau_weighting, cfg.tau_weights, T)
    U_grid = np.asarray(cfg.U_grid, dtype=float)
    tau_grid = np.asarray(cfg.tau_grid, dtype=int)

    beta_hat_t_u_tau = np.zeros((T, len(U_grid), len(tau_grid)), dtype=float)

    for i_u, U_scalar in tqdm(enumerate(U_grid), desc="Attack Rollouts for beta Calibration ... ", unit="WM Cov Setting"):
        history_seed = cfg.base_seed + cfg.history_seed_offset + 100_000 * i_u
        nominal_hist = _run_nominal_rollout(
                plant_data,
                seed=history_seed,
                U_scalar=float(U_scalar),
                detector=None,
            )
        for i_tau, tau in enumerate(tau_grid):
            replay_start_idx = _resolve_replay_start_idx(cfg.replay_start_mode, int(tau), cfg.replay_start_idx)
            I_runs = []
            for m in range(cfg.attack_rollouts):
                attack_seed = cfg.base_seed + cfg.attack_seed_offset + 100_000 * i_u + 1_000 * i_tau + m
                attacked = _run_replay_attack_rollout(
                    plant_data,
                    seed_attack=attack_seed,
                    history_y=nominal_hist["y_hist"],
                    U_scalar=float(U_scalar),
                    tau=int(tau),
                    replay_start_idx=int(replay_start_idx),
                    wrap_history=cfg.wrap_history,
                    detector=detector,
                )
                I_runs.append(attacked["I_hist"])

            I_runs = np.asarray(I_runs, dtype=int)  # (M, T)
            beta_hat_t_u_tau[:, i_u, i_tau] = np.mean(I_runs == 0, axis=0)

    beta_hat_t_u = np.tensordot(beta_hat_t_u_tau, tau_weights, axes=([2], [0]))  # (T, |U|)

    out_path = cfg.out_dir / "beta_lookup.npz"
    np.savez_compressed(
        out_path,
        U_grid=U_grid,
        tau_grid=tau_grid,
        tau_weights=tau_weights,
        g_tilde=np.array(g_tilde, dtype=float),
        alpha=np.array(cfg.alpha, dtype=float),
        beta_hat_t_u_tau=beta_hat_t_u_tau,
        beta_hat_t_u=beta_hat_t_u,
        nominal_g_samples=g_nom_all,
        metadata=json.dumps(
            {
                "plant_data_path": str(cfg.plant_data_path),
                "noise_override": cfg.noise_override,
                "nominal_rollouts": cfg.nominal_rollouts,
                "attack_rollouts": cfg.attack_rollouts,
                "base_seed": cfg.base_seed,
                "history_seed_offset": cfg.history_seed_offset,
                "attack_seed_offset": cfg.attack_seed_offset,
                "replay_start_mode": cfg.replay_start_mode,
                "replay_start_idx": cfg.replay_start_idx,
                "wrap_history": cfg.wrap_history,
            }
        ),
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline calibration of beta-hat lookup for non-Gaussian replay detection.")
    ap.add_argument("--config", type=str, required=True, help="Path to config/msd/calibrate_beta.yaml")
    args = ap.parse_args()

    cfg = _parse_cfg(Path(args.config).expanduser().resolve())
    out_path = calibrate_lookup(cfg)
    print(f"Saved beta lookup to: {out_path}")


if __name__ == "__main__":
    main()
