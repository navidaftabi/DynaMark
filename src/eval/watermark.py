from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def select_U(
    wm_cfg: Dict[str, Any],
    env,
    action: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose watermark covariance U for this decision epoch.

    Modes:
      - none:           U = 0
      - constant_cov:   U = provided matrix
      - from_action:    U = cov_from_action(action)
    """
    mode = str(wm_cfg.get("mode", "none")).lower().strip()
    d = int(getattr(env, "d", getattr(env.plant, "u_dim", 1)))

    if mode == "none":
        return np.zeros((d, d), dtype=float)

    if mode == "constant_cov":
        U = np.asarray(wm_cfg.get("U"), dtype=float)
        if U.shape != (d, d):
            # allow scalar for d=1
            if d == 1 and U.size == 1:
                return np.array([[float(U.reshape(-1)[0])]], dtype=float)
            raise ValueError(f"constant_cov expects U of shape {(d,d)}, got {U.shape}")
        return U

    if mode == "from_action":
        if action is None:
            raise ValueError("watermark.mode='from_action' requires a non-None action")
        from env.core.covariance import cov_from_action  
        return cov_from_action(np.asarray(action, dtype=float), d)

    raise ValueError(f"Unknown watermark.mode='{mode}'")
