
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from .base import PlantBase
from ..core.types import StepOut
from ..core.utils import _to_1d, GMM


class SMDTContinuousPlant(PlantBase):
    """
    Continuous piecewise linear plant for stepper-motor digital twin (scalar).

    True state/output x_t (scalar) evolves as:
      x_{t+1} = A_i x_t + B_i u_applied + w_t
    Observed measurement stream y_t is:
      y_t = x_t (normal)
      y_{t+1} = next_y_override (replay)

    Baseline input u_wowm is sampled from a segment GMM.
    Watermark phi_t ~ N(0, U_t), U_t is (1,1).

    Predictor for residual uses OBSERVED y_t:
      y_hat = A_i y_t + B_i u_cmd

    Residual uses OBSERVED y_{t+1}:
      r = y_{t+1} - y_hat

    Attack hook (when next_y_override is provided):
      - applied control is flipped: u_applied = -u_cmd
      - observed y_{t+1} is overwritten by next_y_override
    """

    y_dim = 1
    u_dim = 1

    def __init__(self, data: Dict[str, Any], *, seed: Optional[int] = None):
        self.data = data
        self._seed = int(0 if seed is None else seed)
        self._rng = np.random.default_rng(self._seed)

        # --- arrays per block (length = num_blocks) ---
        self.A_lst = _to_1d(data["A"], "A")
        self.B_lst = _to_1d(data["B"], "B")
        self.Q_lst = _to_1d(data["Q"], "Q")
        self.y_bar_lst = _to_1d(data["y_bar"], "y_bar")

        self.num_blocks = int(self.A_lst.size)
        if not (self.B_lst.size == self.Q_lst.size == self.y_bar_lst.size == self.num_blocks):
            raise ValueError("A, B, Q, y_bar must have the same length")

        # initial state distribution
        self.mu0 = float(data.get("mu0", 0.0)) 
        self.Sigma0 = float(data.get("Sigma", 1.0))

        # block-to-segment mapping (length num_blocks), values in {0,1,2,3}
        # If not provided, default to zeros. One piece with self.num_blocks blocks.
        block_to_seg = data.get("block_to_seg", None)
        if block_to_seg is None:
            self.block_to_seg = np.zeros(self.num_blocks, dtype=int)
        else:
            self.block_to_seg = np.asarray(block_to_seg, dtype=int).reshape(-1)
            if self.block_to_seg.size != self.num_blocks:
                raise ValueError("block_to_seg must have length = num_blocks")

        # GMMs for each segment 0..3
        # expected: data["ut_gmms"] is dict or list with 4 entries.
        # each entry has keys weights/means/covs
        gmms = data["ut_gmms"]
        self.segment_gmms: Dict[int, GMM] = {}
        for seg_id, seg_cfg in gmms.items():
            self.segment_gmms[int(seg_id)] = GMM(seg_cfg, self._rng)

        for i in range(self.num_blocks):
            if i not in self.segment_gmms:
                raise KeyError(f"Missing segment_gmms for segment {i}")

    @property
    def y_curr(self) -> np.ndarray:
        # current observed measurement as (1,1)
        return np.array([[self._y_obs]], dtype=float)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

        self._t = 0
        self.block_idx = 0
        self._terminate = False

        # sample x0 ~ N(mu0, Sigma0)
        x0 = float(self._rng.normal(loc=self.mu0, scale=np.sqrt(self.Sigma0 + 1e-15)))

        self._x_true = x0
        self._y_obs = x0          # observed starts true
        self._x_wowm = x0         # wowm starts true

        # load first block parameters
        self._set_parameters()

    # --------- block scheduling ----------
    def _set_parameters(self) -> None:
        """
        Load A,B,Q,y_bar and segment GMM for current block_idx.
        Set terminate if past the end.
        """
        i = int(self.block_idx)
        if i >= self.num_blocks:
            self._terminate = True
            return

        self.A = float(self.A_lst[i])
        self.B = float(self.B_lst[i])
        self.Q = float(self.Q_lst[i])
        self.y_bar = float(self.y_bar_lst[i])
        self.seg = int(self.block_to_seg[i])

        # For detector Q is used as (1,1)
        # For predictor Jacobian wrt u: H = B_i (since y_hat = A_i y + B_i u)
        # Segment GMM for baseline inputs:
        self.ut_gmm = self.segment_gmms[self.seg]

    def _advance_block(self) -> None:
        """
        Decide whether to move to next block based on sign of (y_now - y_bar),
        matching your old logic:
          if seg < 2: advance when delta > 0
          else      : advance when delta < 0
        """
        if self._terminate:
            return

        y_now = float(self._y_obs)  # use observed stream like your old code used Y[-1]
        delta = y_now - float(self.y_bar)

        if self.seg < 2:
            if delta > 0:
                self.block_idx += 1
                self._set_parameters()
        else:
            if delta < 0:
                self.block_idx += 1
                self._set_parameters()

    # --------- noise + watermark ----------
    def _sample_w(self) -> float:
        # w ~ N(0, Q)
        return float(self._rng.normal(loc=0.0, scale=np.sqrt(self.Q + 1e-15)))

    def _sample_phi(self, U: np.ndarray) -> float:
        U_scalar = float(np.asarray(U, dtype=float).reshape(1, 1)[0, 0])
        if U_scalar < 0:
            raise ValueError("U must be >= 0 (scalar covariance)")
        return float(self._rng.normal(loc=0.0, scale=np.sqrt(U_scalar)))

    # --------- main step ----------
    def step(self, U: np.ndarray, *, next_y_override=None, flip_on_override: bool = True) -> StepOut:
        """
        One detector step.

        next_y_override semantics:
          - None: normal
          - not None: replay y_{t+1} = next_y_override and flip attack
        """
        U = U * (3.3 / 1000)**2
        if self._terminate:
            # still return something valid; env will terminate
            y_prev = np.array([[self._y_obs]], dtype=float)
            x_prev = np.array([[self._x_true]], dtype=float)
            y_wowm = np.array([[self._x_wowm]], dtype=float)
            z = np.array([[0.0]], dtype=float)
            ctx = {
                "Q": np.array([[self.Q]], dtype=float), 
                "H": np.array([[self.B]], dtype=float),
                "A": np.array([[self.A]], dtype=float),
                "B": np.array([[self.B]], dtype=float),
                }
            return StepOut(
                y_prev=y_prev, y_curr=y_prev, y_wowm_curr=y_wowm,
                x_prev=x_prev, x_curr=x_prev, x_wowm_curr=y_wowm,
                u=z, phi=z, y_hat=y_prev, Q=np.array([[self.Q]], dtype=float), r=z, ctx=ctx,
                terminated=True
            )

        U = np.asarray(U, dtype=float)
        if U.shape != (1, 1):
            raise ValueError(f"SM continuous expects U shape (1,1), got {U.shape}")

        # load current block parameters
        self._advance_block()
        if self._terminate:
            return self.step(U, next_y_override=next_y_override)

        # record prev streams
        y_prev_obs = float(self._y_obs)
        x_prev_true = float(self._x_true)
        x_prev_wowm = float(self._x_wowm)

        y_prev = np.array([[y_prev_obs]], dtype=float)
        x_prev = np.array([[x_prev_true]], dtype=float)

        # sample noises (same w drives true and wowm like your old implementation)
        w_t = self._sample_w()

        # baseline input from segment GMM
        u_wowm = self.ut_gmm.sample()

        # watermark and commanded control
        phi_t = self._sample_phi(U)
        u_cmd = u_wowm + phi_t

        # wowm evolution (true reference)
        x_wowm_next = (self.A * x_prev_wowm) + (self.B * u_wowm) + w_t
        y_wowm_next = x_wowm_next  # measurement equals state in this model

        # predictor uses OBSERVED y_t
        y_hat = (self.A * y_prev_obs) + (self.B * u_cmd)

        # true evolution uses flipped control under replay
        if next_y_override is None:
            u_applied = u_cmd
        else:
            u_applied = -u_cmd if flip_on_override else u_cmd

        x_true_next = (self.A * x_prev_true) + (self.B * u_applied) + w_t

        # observed next measurement (replayed if provided)
        if next_y_override is None:
            y_obs_next = x_true_next
        else:
            y_obs_next = float(np.asarray(next_y_override, dtype=float).reshape(1)[0])

        # residual uses OBSERVED y_{t+1}
        r = y_obs_next - y_hat

        # update internal streams
        self._t += 1
        self._x_true = x_true_next
        self._y_obs = y_obs_next
        self._x_wowm = x_wowm_next

        # prepare outputs as (1,1) matrices
        y_curr = np.array([[y_obs_next]], dtype=float)
        x_curr = np.array([[x_true_next]], dtype=float)
        y_wowm_curr = np.array([[y_wowm_next]], dtype=float)
        x_wowm_curr = np.array([[x_wowm_next]], dtype=float)

        u_mat = np.array([[u_cmd]], dtype=float)
        phi_mat = np.array([[phi_t]], dtype=float)
        y_hat_mat = np.array([[y_hat]], dtype=float)
        r_mat = np.array([[r]], dtype=float)
        Q_mat = np.array([[self.Q]], dtype=float)

        terminated = bool(self._terminate or (self.block_idx >= self.num_blocks))

        ctx = {
            "Q": Q_mat,
            "H": np.array([[self.B]], dtype=float),  # Jacobian wrt u for predictor
            "A": np.array([[self.A]], dtype=float),
            "B": np.array([[self.B]], dtype=float),
            "U_applied": U,
            "block_idx": int(self.block_idx),
            "seg": int(self.seg),
            "t": int(self._t),
            "u_nom": np.array([[u_wowm]], dtype=float),
            "u_applied": np.array([[u_applied]], dtype=float),
            "u_cmd": u_mat
        }

        return StepOut(
            y_prev=y_prev,
            y_curr=y_curr,
            y_wowm_curr=y_wowm_curr,
            x_prev=x_prev,
            x_curr=x_curr,
            x_wowm_curr=x_wowm_curr,
            u=u_mat,
            phi=phi_mat,
            y_hat=y_hat_mat,
            Q=Q_mat,
            r=r_mat,
            ctx=ctx,
            terminated=terminated,
        )
