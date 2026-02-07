
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# plants
from .plants.dt_linear import DigitalTwinLTIPlant
from .plants.msd_nonlinear import MSDNonlinearPlant
from .plants.sm_dt_continuous import SMDTContinuousPlant
from .plants.sm_dt_discrete import SMDTDiscretePlant

# env
from .environment import DynaMarkEnv


@dataclass(frozen=True)
class EnvSpec:
    """
    Everything needed to construct a ready-to-train Gymnasium env.

    Pass:
      - env_name:   "dt" | "msd" | "sm_dt_cont" | "sm_dt_disc" (or aliases)
      - plant_data: already-loaded JSON dict
      - env_cfg:    already-loaded YAML dict
      - seed:       optional RNG seed for plant + env
      - beta_seed:  optional RNG seed for beta-model MC
    """
    env_name: str
    plant_data: Dict[str, Any]
    env_cfg: Dict[str, Any]
    seed: Optional[int] = None
    beta_seed: Optional[int] = None

_DT_ALIASES = {
    "dt", "dt_lti", "digital_twin", "digitaltwin", "lti", "linear_dt"
}
_MSD_ALIASES = {
    "msd", "msd_nl", "msd_nonlinear", "msdnonlinear"
}
_SM_CONT_ALIASES = {
    "sm", "sm_dt_cont", "sm_dt_continuous", "stepper_cont", "sm_cont", "sm_continuous"
}
_SM_DISC_ALIASES = {
    "sm_dt_disc", "sm_dt_discrete", "sm-dt-discrete", "stepper_disc", "sm_disc",
    "sm_dt",
}

def _canonical_env_name(env_name: str) -> str:
    name = str(env_name).lower().strip()
    if name in _DT_ALIASES:
        return "dt"
    if name in _MSD_ALIASES:
        return "msd"
    if name in _SM_CONT_ALIASES:
        return "sm_dt_cont"
    if name in _SM_DISC_ALIASES:
        return "sm_dt_disc"
    raise ValueError(
        f"Unknown env_name='{env_name}'. Expected one of "
        f"dt | msd | sm_dt_cont | sm_dt_disc (or aliases)."
    )

def _normalize_cfg(env_cfg: Dict[str, Any], *, beta_seed: int) -> Dict[str, Any]:
    """
    DynaMarkEnv reads:
      - reward: {w1,w2,w3}
      - detector: {alpha}
      - belief: {q_prior, alpha, p_geom}
      - beta: {alpha, p_geom, window_size, delta_t, n_mc, seed}
      - beta_seed (top-level) for defaults
    """
    cfg = dict(env_cfg) if env_cfg is not None else {}
    cfg.setdefault("beta_seed", int(beta_seed))

    if "reward" not in cfg:
        r = {}
        for k in ("w1", "w2", "w3"):
            if k in cfg:
                r[k] = cfg[k]
        cfg["reward"] = r

    if "detector" not in cfg:
        det = {}
        if "alpha" in cfg:
            det["alpha"] = cfg["alpha"]
        cfg["detector"] = det

    if "belief" not in cfg:
        bel = {}
        if "q_prior" in cfg:
            bel["q_prior"] = cfg["q_prior"]
        elif "q" in cfg:
            bel["q_prior"] = cfg["q"]
        if "alpha" in cfg:
            bel["alpha"] = cfg["alpha"]
        if "p_geom" in cfg:
            bel["p_geom"] = cfg["p_geom"]
        cfg["belief"] = bel

    if "beta" not in cfg:
        beta = {}
        for k in ("alpha", "p_geom", "window_size", "delta_t", "n_mc"):
            if k in cfg:
                beta[k] = cfg[k]
        beta["seed"] = int(cfg.get("beta_seed", beta_seed))
        cfg["beta"] = beta
    else:
        beta = dict(cfg["beta"])
        beta.setdefault("seed", int(cfg.get("beta_seed", beta_seed)))
        cfg["beta"] = beta

    return cfg

def make_plant(
    env_name: str,
    plant_data: Dict[str, Any],
    *,
    env_cfg: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
):
    """
    Create a PlantBase instance
    """
    canon = _canonical_env_name(env_name)
    env_cfg = env_cfg or {}

    if canon == "dt":
        return DigitalTwinLTIPlant(plant_data, seed=seed)

    if canon == "msd":
        return MSDNonlinearPlant(plant_data, seed=seed)

    if canon == "sm_dt_cont":
        return SMDTContinuousPlant(plant_data, seed=seed)

    if canon == "sm_dt_disc":
        plant_block_len = int(env_cfg.get("plant_block_len", env_cfg.get("block_len", 500)))
        proc_len = int(env_cfg.get("proc_len", 100))
        T_fast = env_cfg.get("T_fast", plant_data.get("T_fast", plant_data.get("T", None)))

        return SMDTDiscretePlant(
            plant_data,
            seed=seed,
            plant_block_len=plant_block_len,
            proc_len=proc_len,
            T_fast=T_fast,
        )
    raise ValueError(f"Unhandled canonical env_name='{canon}'")


def make_env(
    env_name: str,
    *,
    plant_data: Dict[str, Any],
    env_cfg: Dict[str, Any],
    seed: int = 0,
    beta_seed: Optional[int] = None,
):
    """
    Construct a ready-to-train DynaMarkEnv(plant, cfg).
    """
    canon = _canonical_env_name(env_name)

    beta_seed_final = int(beta_seed if beta_seed is not None else env_cfg.get("beta_seed", 0))

    plant = make_plant(canon, plant_data, env_cfg=env_cfg, seed=seed)
    cfg = _normalize_cfg(env_cfg, beta_seed=beta_seed_final)

    return DynaMarkEnv(plant=plant, cfg=cfg)


def make_from_spec(spec: EnvSpec):
    return make_env(
        spec.env_name,
        plant_data=spec.plant_data,
        env_cfg=spec.env_cfg,
        seed=int(spec.seed if spec.seed is not None else 0),
        beta_seed=spec.beta_seed,
    )
