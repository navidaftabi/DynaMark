import numpy as np
from scipy.stats import chi2

class ChiSquareDetector:
    def __init__(self, 
                 alpha: float | None = None, 
                 dof: int = 1, 
                 g_tilde: float | None = None):
        self.dof = int(dof)
        
        if alpha is not None and g_tilde is not None:
            raise ValueError("ChiSquareDetector requires either 'alpha' or 'g_tilde'. Not both.")
            
        if alpha is None and g_tilde is None:
            raise ValueError("ChiSquareDetector requires either 'alpha' or 'g_tilde'.")

        self.alpha = None if alpha is None else float(alpha)

        if g_tilde is not None:
            self.g_tilde = float(g_tilde)
        else:
            self.g_tilde = float(chi2.ppf(1.0 - float(alpha), df=self.dof))

    def test(self, r: np.ndarray, Q: np.ndarray) -> tuple[int, float]:
        r = np.atleast_2d(r).reshape(-1, 1)
        Q = np.atleast_2d(Q)
        Qinv = np.linalg.inv(Q)
        g = float((r.T @ Qinv @ r).item())
        It = 0 if g < self.g_tilde else 1
        return It, g