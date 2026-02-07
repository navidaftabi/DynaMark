from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

import json
import numpy as np
import pandas as pd

from env.plants.sm_dt_continuous import SMDTContinuousPlant
from .online import TACOnlineWatermarker

U_SCALE = (3.3 / 1000.0) ** 2

def _as_1d(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def _as_col(a: Any) -> List[List[float]]:
    """Return column-vector nested list [[...],[...]] like src/eval."""
    x = np.asarray(a, dtype=float)
    if x.ndim == 0:
        return [[float(x)]]
    if x.ndim == 1:
        return [[float(v)] for v in x.reshape(-1)]
    if x.ndim == 2:
        if x.shape[1] == 1:
            return [[float(v)] for v in x[:, 0]]
        if x.shape[0] == 1:
            return [[float(v)] for v in x.reshape(-1)]
        return [[float(v) for v in row] for row in x]
    return [[float(v)] for v in x.reshape(-1)]


def _as_mat(a: Any) -> List[List[float]]:
    """Return 2D list for matrices (Q, etc.). Scalar -> [[scalar]]."""
    x = np.asarray(a, dtype=float)
    if x.ndim == 0:
        return [[float(x)]]
    if x.ndim == 1:
        if x.size == 1:
            return [[float(x.reshape(-1)[0])]]
        return [[float(v) for v in x.reshape(-1)]]
    if x.ndim == 2:
        return [[float(v) for v in row] for row in x]
    return [[float(v) for v in x.reshape(-1)]]


def _jsonify(obj: Any) -> Any:
    """Convert numpy types to JSON-safe python types, preserving nested lists."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonify(obj), f, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Match src/eval writer style: pandas records/lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True)


def _complex_array_to_reim(z: Optional[np.ndarray]) -> Dict[str, Any]:
    if z is None:
        return {"re": None, "im": None}
    z = np.asarray(z)
    return {"re": np.real(z).tolist(), "im": np.imag(z).tolist()}

def _calibrate_eta(
    rows: List[Dict[str, Any]],
    *,
    alpha: float,
    burn_in: int,
) -> float:
    """
    eta = quantile_{1-alpha}( g_hat[t >= burn_in] )

    If insufficient finite samples exist, returns +inf (=> no alarms).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("false_alarm_rate alpha must be in (0,1)")

    g_vals: List[float] = []
    for r in rows:
        t = int(r.get("t", 0))
        if t < burn_in:
            continue
        g = r.get("g_hat", None)
        if g is None:
            continue
        try:
            gv = float(g)
        except Exception:
            continue
        if np.isfinite(gv):
            g_vals.append(gv)

    if len(g_vals) < 50:
        return float("inf")

    g_arr = np.asarray(g_vals, dtype=float)
    return float(np.quantile(g_arr, 1.0 - alpha))

def learn_tac_once(
    *,
    out_dir: Path,
    plant_data: Dict[str, Any],
    seed: int,
    steps: int,
    n_eigs: int,
    delta: float,
    beta: float,
    update_interval: int,
    min_k_for_updates: int,
    false_alarm_rate: float = 0.01,
    eta_burn_in: Optional[int] = None,
) -> Dict[str, Any]:
    """Run TAC online learning once (nominal) and save learning trajectory + final_state."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plant = SMDTContinuousPlant(data=plant_data, seed=seed)
    plant.reset(seed=seed)

    tac = TACOnlineWatermarker(
        y_dim=1,
        include_u_dim=1,
        phi_dim=1,
        n_eigs=n_eigs,
        delta=float(delta),
        beta=float(beta),
        update_interval=int(update_interval),
        min_k_for_updates=int(min_k_for_updates),
        Xyy=np.eye(2),
        Xphiphi=np.eye(1),
    )

    learn_rows: List[Dict[str, Any]] = []

    for t in range(int(steps)):
        U_internal = tac.current_U()
        U_env = U_internal / U_SCALE 

        out = plant.step(U_env)

        y_next = _as_1d(out.y_curr)           # (1,)
        u_cmd = _as_1d(out.u)                 # (1,)
        phi = _as_1d(out.phi)                 # (1,)
        u_nom = u_cmd - phi

        U_used = np.asarray(out.ctx.get("U_applied", U_internal), dtype=float)

        info = tac.step(y_next=y_next, u_ctrl_k=u_nom, phi_k=phi, U_k=U_used)

        roots_reim = _complex_array_to_reim(info.get("roots", None))
        Omega = info.get("Omega", None)
        Omega_reim = _complex_array_to_reim(Omega) if Omega is not None else {"re": None, "im": None}

        g_hat = info["g_hat"]
        learn_rows.append(
            {
                "t": int(t),
                "terminated": bool(out.terminated),
                "y": float(y_next[0]),
                "u_cmd": float(u_cmd[0]),
                "u_nom": float(u_nom[0]),
                "phi": float(phi[0]),
                "U_internal": float(U_internal.reshape(-1)[0]),
                "U_env": float(U_env.reshape(-1)[0]),
                "U_used": float(np.asarray(U_used).reshape(-1)[0]),
                "U_star": float(np.asarray(info["U_star"]).reshape(-1)[0]),
                "U_next": float(np.asarray(info["U_next"]).reshape(-1)[0]),
                "g_hat": float(g_hat) if np.isfinite(g_hat) else None,
                "resid": _jsonify(info.get("resid", None)),
                "roots": roots_reim,
                "Omega": Omega_reim,
                "W_hat": _jsonify(info["W_hat"]),
                "Uphi_hat": _jsonify(info["Uphi_hat"]),
                "P_hat": _jsonify(info["P_hat"]),
                "X_hat": _jsonify(info["X_hat"]),
                "ctx": _jsonify(out.ctx),
            }
        )

        if out.terminated:
            break

    _write_jsonl(out_dir / "learn_traj.jsonl", learn_rows)

    if eta_burn_in is None:
        eta_burn_in = max(int(min_k_for_updates), int(5 * update_interval))
    eta = _calibrate_eta(learn_rows, alpha=float(false_alarm_rate), burn_in=int(eta_burn_in))

    final = learn_rows[-1] if learn_rows else {}
    final_state = {
        "delta": float(delta),
        "seed": int(seed),
        "steps_ran": int(len(learn_rows)),
        "U_star": float(final.get("U_star", np.nan)),
        "U_internal_last": float(final.get("U_internal", np.nan)),
        "W_hat": final.get("W_hat", None),
        "Uphi_hat": final.get("Uphi_hat", None),
        "roots": final.get("roots", None),
        "Omega": final.get("Omega", None),
        "false_alarm_rate": float(false_alarm_rate),
        "eta_burn_in": int(eta_burn_in),
        "eta": float(eta),
        "notes": {
            "U_SCALE": float(U_SCALE),
            "U_units": "U_star is TAC-internal (post scaling). Use U_env = U_star / U_SCALE for plant.step().",
            "eta_rule": "Alarm if g_hat > eta.",
        },
    }
    _write_json(out_dir / "final_state.json", final_state)
    return final_state

def collect_nominal_y_history(
    *,
    out_dir: Path,
    plant_data: Dict[str, Any],
    seed: int,
    steps: int,
    U_star_internal: float,
) -> np.ndarray:
    """Generate and save full nominal y[t] history (exact indices) for replay."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plant = SMDTContinuousPlant(data=plant_data, seed=seed)
    plant.reset(seed=seed)

    U_star = np.array([[float(U_star_internal)]], dtype=float)
    U_env = U_star / U_SCALE

    y_hist: List[float] = []
    for _t in range(int(steps)):
        out = plant.step(U_env)
        y_hist.append(float(np.asarray(out.y_curr).reshape(-1)[0]))
        if out.terminated:
            break

    y_arr = np.asarray(y_hist, dtype=float)
    _write_json(out_dir / "nominal_history.json", {"seed": int(seed), "U_star": float(U_star_internal), "y": y_arr})
    return y_arr


def simulate_rep(
    *,
    out_dir: Path,
    plant_data: Dict[str, Any],
    seed: int,
    steps: int,
    rep: int,
    eta: float,
    U_star_internal: float,
    learned_W_hat: Optional[Any],
    learned_Uphi_hat: Optional[Any],
    learned_roots: Optional[Dict[str, Any]],
    learned_Omega: Optional[Dict[str, Any]],
    n_eigs: int,
    # replay attack params
    attack: bool,
    y_replay_hist: Optional[np.ndarray],
    attack_onset: int,
    attack_len: int,
    replay_from: int,
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)

    plant = SMDTContinuousPlant(data=plant_data, seed=seed)
    plant.reset(seed=seed)

    tac = TACOnlineWatermarker(
        y_dim=1,
        include_u_dim=1,
        phi_dim=1,
        n_eigs=n_eigs,
        delta=1.0,
        beta=1 / 3,
        update_interval=10**9,
        min_k_for_updates=10**9,
        Xyy=np.eye(2),
        Xphiphi=np.eye(1),
    )

    U_star = np.array([[float(U_star_internal)]], dtype=float)
    tac.U_star = U_star.copy()
    tac.U = U_star.copy()

    if learned_W_hat is not None:
        tac.W_hat = np.asarray(learned_W_hat, dtype=float)
    if learned_Uphi_hat is not None:
        tac.Uphi_hat = np.asarray(learned_Uphi_hat, dtype=float)

    if learned_roots and learned_roots.get("re") is not None:
        re = np.asarray(learned_roots["re"], dtype=float)
        im = np.asarray(learned_roots["im"], dtype=float)
        tac.roots = re + 1j * im
    if learned_Omega and learned_Omega.get("re") is not None:
        re = np.asarray(learned_Omega["re"], dtype=float)
        im = np.asarray(learned_Omega["im"], dtype=float)
        tac.Omega = re + 1j * im

    U_env = U_star / U_SCALE

    traj_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    It_list: List[int] = []
    g_list: List[float] = []

    BIG = 10**18
    onset_t = int(attack_onset) if attack else BIG
    atk_type = "replay" if attack else "none"

    for k in range(int(steps)):
        next_y_override = None
        attack_active = False

        if attack:
            if y_replay_hist is None:
                raise ValueError("attack=True requires y_replay_hist")
            if attack_onset <= k < (attack_onset + attack_len):
                attack_active = True
                idx = int(replay_from) + (k - int(attack_onset))
                if idx < 0 or idx >= len(y_replay_hist):
                    raise ValueError(f"Replay index out of range: idx={idx}, len(hist)={len(y_replay_hist)}")
                next_y_override = float(y_replay_hist[idx])

        out = plant.step(U_env, next_y_override=next_y_override)

        # plant outputs
        y_next = _as_1d(out.y_curr)      # (1,)
        u_cmd = _as_1d(out.u)            # (1,)
        phi = _as_1d(out.phi)            # (1,)
        u_nom = u_cmd - phi

        U_used = np.asarray(out.ctx.get("U_applied", U_star), dtype=float)  # post-scale
        U_var = float(U_used.reshape(-1)[0])
        action_std = float(np.sqrt(max(U_var, 0.0)))

        info = tac.step(y_next=y_next, u_ctrl_k=u_nom, phi_k=phi, U_k=U_used)

        # freeze U
        tac.U_star = U_star.copy()
        tac.U = U_star.copy()

        g_hat = float(info["g_hat"]) if np.isfinite(info["g_hat"]) else float("nan")
        g_list.append(g_hat)

        # alarm
        It = 1 if (np.isfinite(g_hat) and (g_hat > float(eta))) else 0
        It_list.append(It)

        # costs
        phi_cost = float(abs(phi.reshape(-1)[0]))
        # delta_cost
        try:
            x_true = np.asarray(out.x_curr, dtype=float)
            x_wowm = np.asarray(out.x_wowm_curr, dtype=float)
            delta_cost = float(np.linalg.norm((x_wowm - x_true).reshape(-1), ord=2))
        except Exception:
            delta_cost = 0.0

        # residual vector
        try:
            r_col = _as_col(out.r)
        except Exception:
            r_col = [[float("nan")]]

        row = {
            "rep": int(rep),
            "k": int(k),
            "t_dec": int(k + 1),
            "attack_active": bool(attack_active),
            "attack_type": str(atk_type),
            "onset_t": int(onset_t),
            "y": _as_col(out.y_curr),
            "S1": float("nan"),
            "U": [float(U_var)],
            "action": [float(action_std)],
            "terminated": bool(out.terminated),
            "batched": False,
            "It": int(It),
            "g": float(g_hat),
            "beta": 1.0,
            "reward": 0.0,
            "phi_cost": float(phi_cost),
            "delta_cost": float(delta_cost),
            "info_bonus": 0.0,
            "x_true": _as_col(out.x_curr),
            "x_wowm": _as_col(out.x_wowm_curr),
            "y_hat": _as_col(out.y_hat),
            "r": r_col,
            "u": _as_col(out.u),
            "phi": _as_col(out.phi),
            "Q": _as_mat(out.Q),
        }
        traj_rows.append(row)

        diag_rows.append(
            {
                "rep": int(rep),
                "k": int(k),
                "next_y_override": next_y_override,
                "u_cmd": float(u_cmd[0]),
                "u_nom": float(u_nom[0]),
                "U_env": float(U_env.reshape(-1)[0]),
                "U_used": float(U_var),
                "ctx": _jsonify(out.ctx),
            }
        )

        if out.terminated:
            break

    # write files
    _write_jsonl(out_dir / "traj.jsonl", traj_rows)
    _write_jsonl(out_dir / "diag.jsonl", diag_rows)

    It_arr = np.asarray(It_list, dtype=float)
    g_arr = np.asarray([x for x in g_list if np.isfinite(x)], dtype=float)

    pre = It_arr[:attack_onset] if attack_onset < len(It_arr) else It_arr
    post = It_arr[attack_onset:] if attack_onset < len(It_arr) else np.asarray([], dtype=float)

    summary = {
        "seed": int(seed),
        "rep": int(rep),
        "steps_ran": int(len(traj_rows)),
        "attack": bool(attack),
        "U_star": float(U_star_internal),
        "eta": float(eta),
        "g_mean": float(np.mean(g_arr)) if g_arr.size else None,
        "g_max": float(np.max(g_arr)) if g_arr.size else None,
        "alarm_rate": float(np.mean(It_arr)) if It_arr.size else None,
        "alarm_rate_pre": float(np.mean(pre)) if pre.size else None,
        "alarm_rate_post": float(np.mean(post)) if post.size else None,
        "attack_onset": int(attack_onset),
        "attack_len": int(attack_len),
        "replay_from": int(replay_from),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary
