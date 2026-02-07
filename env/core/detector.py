import numpy as np
from scipy.stats import chi2

class ChiSquareDetector:
    def __init__(self, alpha: float, dof: int):
        self.alpha = float(alpha)
        self.dof = int(dof)
        self.gamma = float(chi2.ppf(1.0 - self.alpha, df=self.dof))

    def test(self, r: np.ndarray, Q: np.ndarray) -> tuple[int, float]:
        r = np.atleast_2d(r).reshape(-1, 1)
        Q = np.atleast_2d(Q)
        Qinv = np.linalg.inv(Q)
        g = float((r.T @ Qinv @ r).item())
        It = 0 if g < self.gamma else 1
        return It, g
