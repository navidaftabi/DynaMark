from dataclasses import dataclass
from scipy.stats import geom

@dataclass
class BeliefState:
    S0: float
    S1: float

class ReplayBeliefFilter:
    """
    Common belief recursion:
      P(I_t | Sigma=0) uses alpha
      P(I_t | Sigma=1) uses beta_t mixed with geometric onset cdf
    """
    def __init__(self, q_prior: float, alpha: float, p_geom: float):
        self.q_prior = float(q_prior)
        self.alpha = float(alpha)
        self.p_geom = float(p_geom)
        self.reset()

    def reset(self):
        self.t = 1
        self.S0_hist = [1.0 - self.q_prior]
        self.S1_hist = [self.q_prior]

    @property
    def S1(self) -> float:
        return self.S1_hist[-1]

    def update(self, It: int, beta_t: float):
        onset_prob = float(geom.cdf(self.t, self.p_geom))

        if It == 0:
            P_It_S0 = 1.0 - self.alpha
            P_It_S1 = beta_t * onset_prob + (1.0 - self.alpha) * (1.0 - onset_prob)
        else:
            P_It_S0 = self.alpha
            P_It_S1 = (1.0 - beta_t) * onset_prob + self.alpha * (1.0 - onset_prob)

        prev_S0 = self.S0_hist[-1]
        prev_S1 = self.S1_hist[-1]
        num0 = P_It_S0 * prev_S0
        num1 = P_It_S1 * prev_S1
        denom = num0 + num1

        self.S0_hist.append(num0 / denom)
        self.S1_hist.append(num1 / denom)
        self.t += 1
