from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class StepOut:
    # observations
    y_prev: np.ndarray
    y_curr: np.ndarray
    y_wowm_curr: np.ndarray

    # system
    x_prev: np.ndarray
    x_curr: np.ndarray
    x_wowm_curr: np.ndarray

    # inputs
    u: np.ndarray
    phi: np.ndarray

    # predictor + residual stats inputs
    y_hat: np.ndarray
    Q: np.ndarray
    r: np.ndarray

    # optional context for beta-models (A,B,H, etc.)
    ctx: Dict[str, Any]

    terminated: bool = False
