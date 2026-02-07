from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "env").is_dir():
            return p
    return start


def jsonify(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    return x


def as_1d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1)


def as_col(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a
    if a.ndim == 2 and a.shape[0] == 1:
        return a.reshape(-1, 1)
    raise ValueError(f"expected scalar/1d/col, got {a.shape}")


def reward_terms(w1: float, w2: float, w3: float, phi: Any, xdiff: Any, S1: float) -> Dict[str, float]:
    phi_cost = float(np.sum(np.abs(np.asarray(phi, dtype=float))))
    delta_cost = float(np.linalg.norm(np.asarray(xdiff, dtype=float)))
    eps = 1e-12
    p = float(np.clip(S1, eps, 1.0 - eps))
    info_bonus = float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))
    reward = float(w1 * phi_cost - w2 * delta_cost + w3 * info_bonus)
    return {
        "reward": reward,
        "phi_cost": phi_cost,
        "delta_cost": delta_cost,
        "info_bonus": info_bonus,
    }


def tag_watermark(wm: Dict[str, Any]) -> str:
    mode = str(wm.get("mode", "none")).lower()
    if mode == "none":
        return "no_wm"
    if mode == "constant_cov":
        U = wm.get("U", [[0.0]])
        try:
            val = float(U[0][0])
        except Exception:
            val = 0.0
        return f"const_cov_{val}"
    if mode == "from_action":
        return "from_action"
    if mode == "random_diag":
        return "rand_diag"
    return mode


def tag_attack(attack: Dict[str, Any]) -> str:
    atype = str(attack.get("type", "none")).lower()
    if atype in ("none", "normal"):
        return "normal"
    onset = attack.get("onset_t", None)
    if onset is None:
        return f"attack_{atype}"
    return f"attack_{atype}_on{int(onset)}"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonify(obj), f, indent=indent)
