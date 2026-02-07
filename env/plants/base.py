import numpy as np
from typing import Optional
from ..core.types import StepOut

class PlantBase:
    y_dim: int
    u_dim: int

    @property
    def y_curr(self): 
        raise NotImplementedError

    def reset(self, seed=None): 
        raise NotImplementedError

    def step(self, U: np.ndarray, *, next_y_override=None) -> StepOut:
        """
        Advance one detector time step with watermark covariance U.
        If next_y_override is provided, replicate replay behavior.
        """
        raise NotImplementedError
