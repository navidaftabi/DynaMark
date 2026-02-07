
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def jsonify(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def write_json(path: Path, obj: Dict[str, Any], *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(jsonify(obj), f, indent=indent)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(jsonify(row)) + "\n")


def _col(v: float) -> list:
    """Return [[v]] to match eval's column-vector style."""
    return [[float(v)]]


def make_row(
    *,
    rep: int,
    k: int,
    t_dec: int,
    attack_active: bool,
    attack_type: str,
    onset_t: int,
    y: list,
    S1: float,
    U: list,
    action: list,
    terminated: bool,
    batched: bool,
    It: int,
    g: float,
    beta: float,
    reward: float,
    phi_cost: float,
    delta_cost: float,
    info_bonus: float,
    x_true: list,
    x_wowm: list,
    y_hat: list,
    r: list,
    u: list,
    phi: list,
    Q: Optional[list] = None,
) -> Dict[str, Any]:
    return {
        "rep": int(rep),
        "k": int(k),
        "t_dec": int(t_dec),
        "attack_active": bool(attack_active),
        "attack_type": str(attack_type),
        "onset_t": int(onset_t),
        "y": y,
        "S1": float(S1),
        "U": U,
        "action": action,
        "terminated": bool(terminated),
        "batched": bool(batched),
        "It": int(It),
        "g": float(g),
        "beta": float(beta),
        "reward": float(reward),
        "phi_cost": float(phi_cost),
        "delta_cost": float(delta_cost),
        "info_bonus": float(info_bonus),
        "x_true": x_true,
        "x_wowm": x_wowm,
        "y_hat": y_hat,
        "r": r,
        "u": u,
        "phi": phi,
        "Q": Q,
    }
