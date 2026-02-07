from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import as_col, jsonify, reward_terms


def advance_one(
    env,
    *,
    U: np.ndarray,
    next_y_override: Optional[Any] = None,
) -> Tuple[np.ndarray, float, bool, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Advance one decision epoch.

    Returns:
      obs_next, reward_total, terminated, decision_info, ctx, processed_rows

    Notes:
      - decision_info is always one dict per decision epoch
      - processed_rows is empty for non-batched plants
      - for sm_dt_disc (batched), processed_rows has proc_len rows
    """
    env.t_dec += 1

    out = env.plant.step(U, next_y_override=next_y_override)
    ctx = getattr(out, "ctx", {}) or {}

    w1 = float(getattr(env, "w1", env.cfg.get("w1", 1.0)))
    w2 = float(getattr(env, "w2", env.cfg.get("w2", 1.0)))
    w3 = float(getattr(env, "w3", env.cfg.get("w3", 0.0)))

    processed_rows: List[Dict[str, Any]] = []

    # batched (SM-DT discrete)
    if "batch" in ctx and ctx["batch"]:
        batch = ctx["batch"]
        L = int(batch.get("proc_len", 0))
        if L <= 0:
            obs = env._get_obs()
            return obs, 0.0, True, {"batched": True, "reason": "empty_batch"}, ctx, processed_rows

        y = np.asarray(batch.get("y", np.zeros(L)), dtype=float).reshape(L)
        y_hat = np.asarray(batch.get("y_hat", np.zeros(L)), dtype=float).reshape(L)
        u = np.asarray(batch.get("u", np.zeros(L)), dtype=float).reshape(L)
        x = np.asarray(batch["x_curr"], dtype=float).reshape(L)
        xw = np.asarray(batch["xw_curr"], dtype=float).reshape(L)
        phi = np.asarray(batch["phi"], dtype=float).reshape(L)
        r = np.asarray(batch["r"], dtype=float).reshape(L)
        Q = np.asarray(batch["Q"], dtype=float).reshape(L)
        B = np.asarray(batch["B"], dtype=float).reshape(L)
        attacked = np.asarray(batch.get("attacked", np.zeros(L)), dtype=int).reshape(L)

        It_seq = np.zeros(L, dtype=int)
        g_seq = np.zeros(L, dtype=float)
        beta_seq = np.full(L, np.nan, dtype=float)
        S1_seq = np.full(L, np.nan, dtype=float)
        S1_prior = float(getattr(env.belief, "S1", 0.0))
        S1_seq[0] = S1_prior

        rt0 = reward_terms(w1, w2, w3, phi[0], xw[0] - x[0], S1_prior)
        processed_rows.append(
            {
                "i": 0,
                "y": float(y[0]),
                "y_hat": float(y_hat[0]),
                "u": float(u[0]),
                "phi": float(phi[0]),
                "x_true": float(x[0]),
                "x_wowm": float(xw[0]),
                "r": float(r[0]),
                "Q": float(Q[0]),
                "B": float(B[0]),
                "attacked": int(attacked[0]),
                "It": int(It_seq[0]),
                "g": float(g_seq[0]),
                "beta": None,
                "S1": float(S1_seq[0]),
                **rt0,
            }
        )

        for i in range(1, L):
            Q_i = float(Q[i])
            r_i = float(r[i])
            It_i, g_i = env.detector.test(as_col(r_i), np.array([[Q_i]], dtype=float))
            It_seq[i] = int(It_i)
            g_seq[i] = float(g_i)

            step_ctx_i = {
                "B": np.array([[float(B[i - 1])]], dtype=float),
                "H": np.array([[float(B[i - 1])]], dtype=float),
                "Q": np.array([[float(Q[i - 1])]], dtype=float),
            }
            env.beta_model.push(U=U, step_ctx=step_ctx_i)
            beta_i = float(env.beta_model.beta_t(t_det=int(getattr(env.belief, "t", 0))))
            beta_seq[i] = beta_i

            env.belief.update(It_i, beta_i)
            S1_i = float(getattr(env.belief, "S1", 0.0))
            S1_seq[i] = S1_i

            rti = reward_terms(w1, w2, w3, phi[i], xw[i] - x[i], S1_i)
            processed_rows.append(
                {
                    "i": int(i),
                    "y": float(y[i]),
                    "y_hat": float(y_hat[i]),
                    "u": float(u[i]),
                    "phi": float(phi[i]),
                    "x_true": float(x[i]),
                    "x_wowm": float(xw[i]),
                    "r": float(r[i]),
                    "Q": float(Q[i]),
                    "B": float(B[i]),
                    "attacked": int(attacked[i]),
                    "It": int(It_seq[i]),
                    "g": float(g_seq[i]),
                    "beta": float(beta_seq[i]),
                    "S1": float(S1_seq[i]),
                    **rti,
                }
            )

        reward_total = float(sum(row["reward"] for row in processed_rows))
        obs = env._get_obs()
        terminated = bool(getattr(out, "terminated", False))
        decision_info = {
            "batched": True,
            "proc_len": int(L),
            "reward_total": float(reward_total),
            "phi_cost_total": float(sum(row["phi_cost"] for row in processed_rows)),
            "delta_cost_total": float(sum(row["delta_cost"] for row in processed_rows)),
            "info_bonus_total": float(sum(row["info_bonus"] for row in processed_rows)),
            "S1_end": float(S1_seq[-1]),
            "epoch_attacked": bool(np.any(attacked)),
            "t_fast_start": ctx.get("t_fast_start", None),
            "n_fast": ctx.get("n_fast", None),
        }
        return obs, reward_total, terminated, decision_info, ctx, processed_rows

    # normal (single-step)
    It, g = env.detector.test(out.r, out.Q)
    env.beta_model.push(U=U, step_ctx=ctx)
    beta_t = float(env.beta_model.beta_t(t_det=int(getattr(env.belief, "t", 0))))
    env.belief.update(It, beta_t)
    S1 = float(getattr(env.belief, "S1", 0.0))

    rt = reward_terms(w1, w2, w3, out.phi, out.x_wowm_curr - out.x_curr, S1)
    obs = env._get_obs()
    terminated = bool(getattr(out, "terminated", False))

    decision_info = {
        "batched": False,
        "It": int(It),
        "g": float(g),
        "beta": float(beta_t),
        "S1": float(S1),
        **rt,
    }
    decision_info.update(
        {
            "x_true": jsonify(out.x_curr),
            "x_wowm": jsonify(out.x_wowm_curr),
            "y": jsonify(out.y_curr),
            "y_hat": jsonify(out.y_hat),
            "r": jsonify(out.r),
            "u": jsonify(out.u),
            "phi": jsonify(out.phi),
            "Q": jsonify(out.Q),
        }
    )

    return obs, float(rt["reward"]), terminated, decision_info, ctx, processed_rows
