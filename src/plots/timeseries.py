#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_json(path, lines=True)


def _rep_dirs(run_dir: Path) -> List[Path]:
    run_dir = Path(run_dir)
    if (run_dir / "traj_decision.jsonl").exists() or (run_dir / "traj_processed.jsonl").exists():
        return [run_dir]
    reps = sorted([p for p in run_dir.glob("rep*") if p.is_dir()])
    if not reps:
        raise FileNotFoundError(f"No rep directories found under {run_dir}")
    return reps


def _infer_alpha(run_dir: Path, default: float = 0.01) -> float:
    cand = run_dir / "run_summary.json"
    if cand.exists():
        try:
            obj = json.loads(cand.read_text())
            cfg = obj.get("config", {})
            env_cfg = (((cfg.get("env") or {}).get("env_cfg")) or {})
            if isinstance(env_cfg.get("detector"), dict) and "alpha" in env_cfg["detector"]:
                return float(env_cfg["detector"]["alpha"])
            if "alpha" in env_cfg:
                return float(env_cfg["alpha"])
            if isinstance(env_cfg.get("belief"), dict) and "alpha" in env_cfg["belief"]:
                return float(env_cfg["belief"]["alpha"])
        except Exception:
            pass
    return float(default)

def _pick_time_col(df: pd.DataFrame) -> str:
    for c in ("t_proc", "t_fast", "t", "t_dec", "k"):
        if c in df.columns:
            return c
    raise KeyError("No time column found. Expected one of: t_proc, t_fast, t, t_dec, k")

def _find_first_listlike(df: pd.Series):
    for v in df:
        if v is None:
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return v
    return None


def _expand_vector(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, int]:
    if col not in df.columns:
        return pd.DataFrame(index=df.index), 0

    dim = None
    for v in df[col]:
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        flat = arr.reshape(-1)
        dim = int(flat.size)
        break

    if dim is None or dim == 0:
        return pd.DataFrame(index=df.index), 0

    rows = []
    for v in df[col]:
        if v is None:
            rows.append([np.nan] * dim)
            continue
        flat = np.asarray(v, dtype=float).reshape(-1)
        if flat.size != dim:
            raise ValueError(f"Column '{col}' has inconsistent sizes: expected {dim}, got {flat.size}")
        rows.append(flat.tolist())

    mat = np.asarray(rows, dtype=float)          # (N, dim)
    out = pd.DataFrame(mat, columns=[f"{col}_{i}" for i in range(dim)], index=df.index)
    return out, dim


def _expand_square_matrix(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, int]:
    if col not in df.columns:
        return pd.DataFrame(index=df.index), 0
    d = None
    for v in df[col]:
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        flat = arr.reshape(-1)
        if flat.size == 0:
            continue
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            d = int(arr.shape[0])
            break
        dd = int(round(np.sqrt(flat.size)))
        if dd * dd == flat.size:
            d = dd
            break

    if d is None:
        return pd.DataFrame(index=df.index), 0

    rows = []
    for v in df[col]:
        if v is None:
            rows.append([np.nan] * (d * d))
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 2:
            if arr.shape != (d, d):
                raise ValueError(f"{col}: expected shape {(d,d)}, got {arr.shape}")
            mat = arr
        else:
            flat = arr.reshape(-1)
            if flat.size != d * d:
                raise ValueError(f"{col}: expected length {d*d}, got {flat.size}")
            mat = flat.reshape(d, d)
        rows.append(mat.reshape(-1).tolist())

    mat = np.asarray(rows, dtype=float)  # (N, d*d)
    cols = [f"{col}_{i}_{j}" for i in range(d) for j in range(d)]
    out = pd.DataFrame(mat, columns=cols, index=df.index)
    return out, d

def _align_reps(dfs: List[pd.DataFrame], time_col: str, cols: List[str]) -> pd.DataFrame:
    wide = None
    for j, df in enumerate(dfs):
        sub = df[[time_col] + cols].drop_duplicates(subset=[time_col]).set_index(time_col).sort_index()
        sub = sub.rename(columns={c: f"{c}__rep{j}" for c in cols})
        wide = sub if wide is None else wide.join(sub, how="inner")
    if wide is None:
        raise RuntimeError("No dataframes to align.")
    return wide.sort_index()


def _stats_from_wide(wide: pd.DataFrame, base_col: str, q_lo: float, q_hi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rep_cols = [c for c in wide.columns if c.startswith(base_col + "__rep")]
    X = wide[rep_cols].to_numpy(dtype=float)  # (T, R)
    mean = np.nanmean(X, axis=1)
    lo = np.nanquantile(X, q_lo, axis=1)
    hi = np.nanquantile(X, q_hi, axis=1)
    return mean, lo, hi

def _plot_band(ax, t, mean, lo, hi, label: str):
    line, = ax.plot(t, mean, label=label)
    ax.fill_between(t, lo, hi, alpha=0.2, color=line.get_color())

def _vline_attack_onset(ax, df_all: pd.DataFrame, time_col: str):
    if "attack_type" in df_all.columns and df_all["attack_type"].notna().any():
        at = str(df_all["attack_type"].dropna().iloc[0]).lower()
        if at in ("none", "normal"):
            return
        
    if "attack_active" in df_all.columns and df_all["attack_active"].notna().any():
        s = df_all["attack_active"]

        if s.dtype == bool:
            act = s.fillna(False)
        else:
            ss = s.astype(str).str.strip().str.lower()
            act = ss.isin(["1", "true", "t", "yes", "y"])

        if act.any():
            t0 = df_all.loc[act, time_col].min()
            if pd.notna(t0):
                ax.axvline(float(t0), linestyle="--")
            return

    if "onset_t" not in df_all.columns or not df_all["onset_t"].notna().any():
        return

    onset_t = int(df_all["onset_t"].dropna().iloc[0])

    if time_col == "t_proc":
        if "proc_len" in df_all.columns and df_all["proc_len"].notna().any():
            L = int(df_all["proc_len"].dropna().iloc[0])
        elif "i" in df_all.columns and df_all["i"].notna().any():
            L = int(df_all["i"].max()) + 1
        else:
            return
        x_on = onset_t * L

    elif time_col in ("k", "t_dec"):
        x_on = onset_t

    elif time_col == "t_fast":
        if "t_fast_start" in df_all.columns and df_all["t_fast_start"].notna().any() and "k" in df_all.columns:
            starts = df_all.loc[df_all["k"] == onset_t, "t_fast_start"]
            if starts.notna().any():
                x_on = float(starts.dropna().iloc[0])
            else:
                return
        else:
            return
    else:
        return

    tmin = float(df_all[time_col].min())
    tmax = float(df_all[time_col].max())
    if tmin <= x_on <= tmax:
        ax.axvline(float(x_on), linestyle="--")

def _square_fig(nrows: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 6), sharex=True)
    if nrows == 1:
        axes = [axes]
    return fig, axes

def _parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y"])


def _alarm_fraction_for_rep(df: pd.DataFrame, *, time_col: str, label: str) -> Tuple[int, int]:
    if "It" not in df.columns:
        return 0, 0

    it = pd.to_numeric(df["It"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = int(len(it))
    if n == 0:
        return 0, 0

    label_l = label.lower()
    is_attack = label_l.startswith("attack")
    if not is_attack:
        return int(np.nansum(it)), n

    mask = None
    if "attack_active" in df.columns:
        act = _parse_bool_series(df["attack_active"])
        if bool(act.any()):
            mask = act.to_numpy(dtype=bool)

    if mask is None:
        if "onset_t" in df.columns and time_col in df.columns:
            try:
                onset_t = int(pd.to_numeric(df["onset_t"], errors="coerce").dropna().iloc[0])
                t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
                mask = t >= float(onset_t)
            except Exception:
                mask = np.ones(n, dtype=bool)
        else:
            mask = np.ones(n, dtype=bool)

    denom = int(np.sum(mask))
    if denom <= 0:
        return 0, 0
    return int(np.nansum(it[mask])), denom


def make_plots_for_condition(
    *,
    run_dir: Path,
    label: str,
    out_dir: Path,
    q_lo: float,
    q_hi: float,
    alpha: Optional[float] = None,
    prefer_processed: str = "auto",  # "auto" | "processed" | "decision"
):
    rep_dirs = _rep_dirs(run_dir)

    # choose which jsonl to load
    def pick_file(rep: Path) -> Path:
        proc = rep / "traj_processed.jsonl"
        dec = rep / "traj_decision.jsonl"
        if prefer_processed == "processed":
            return proc if proc.exists() else dec
        if prefer_processed == "decision":
            return dec
        # auto
        return proc if proc.exists() else dec

    dfs = []
    dfs_dec = []
    for rep in rep_dirs:
        p_main = pick_file(rep)
        dfs.append(_read_jsonl(p_main))
        p_dec = rep / "traj_decision.jsonl"
        dfs_dec.append(_read_jsonl(p_dec) if p_dec.exists() else _read_jsonl(p_main))
    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    df_all_dec = pd.concat(dfs_dec, ignore_index=True, sort=False)

    time_col = _pick_time_col(dfs[0])
    time_col_dec = _pick_time_col(dfs_dec[0])
    if alpha is None:
        alpha = _infer_alpha(run_dir, default=0.01)

    def require_any(keys: List[str], purpose: str) -> str:
        for k in keys:
            if k in dfs[0].columns:
                return k
        raise KeyError(
            f"Missing {purpose}. None of {keys} found in {run_dir} jsonl.\n"
            f"If this is dt/msd/sm_dt_cont: you must resimulate with simulate.py patched to record StepOut fields."
        )

    y_key = require_any(["y"], "measurement y")
    yhat_key = require_any(["y_hat"], "prediction/estimate y_hat")
    x_key = require_any(["x_true", "x"], "true trajectory (x_true or x)")
    u_key = require_any(["u"], "control u")
    r_key = require_any(["r"], "residual r")
    g_key = require_any(["g"], "test statistic g")
    S1_key = require_any(["S1"], "belief S1")

    expanded = []
    dims: Dict[str, int] = {}

    for key in [x_key, y_key, yhat_key, u_key, r_key]:
        ex, dim = _expand_vector(df_all, key)
        expanded.append(ex)
        dims[key] = dim

    g_df, _ = _expand_vector(df_all, g_key)
    S1_df, _ = _expand_vector(df_all, S1_key)

    base = df_all[[time_col, "rep"]].copy() if "rep" in df_all.columns else df_all[[time_col]].copy()
    if "rep" not in base.columns:
        pass

    df_scalar = pd.concat([df_all[[time_col]], *expanded, g_df, S1_df], axis=1)

    rep_scalar_dfs = []
    for df in dfs:
        part = pd.DataFrame({time_col: df[time_col].values})
        for key in [x_key, y_key, yhat_key, u_key, r_key]:
            ex, _ = _expand_vector(df, key)
            part = pd.concat([part, ex], axis=1)
        exg, _ = _expand_vector(df, g_key)
        exS1, _ = _expand_vector(df, S1_key)
        part = pd.concat([part, exg, exS1], axis=1)
        rep_scalar_dfs.append(part)

    rep_wm_dfs = []
    U_dim_raw = 0
    U_dim_app = 0
    phi_dim = 0
    It_present = False
    for df_dec in dfs_dec:
        if time_col_dec in df_dec.columns:
            part = pd.DataFrame({time_col_dec: df_dec[time_col_dec].values})
        else:
            part = pd.DataFrame({time_col_dec: np.arange(len(df_dec), dtype=int)})
        exU, dU = _expand_square_matrix(df_dec, "U")
        if dU > 0:
            if U_dim_raw == 0:
                U_dim_raw = dU
            elif dU != U_dim_raw:
                raise ValueError(f"Inconsistent U dimension across reps: expected {U_dim_raw}, got {dU}")
            part = pd.concat([part, exU], axis=1)

        exUa, dUa = _expand_square_matrix(df_dec, "U_applied")
        if dUa > 0:
            if U_dim_app == 0:
                U_dim_app = dUa
            elif dUa != U_dim_app:
                raise ValueError(f"Inconsistent U_applied dimension across reps: expected {U_dim_app}, got {dUa}")
            part = pd.concat([part, exUa], axis=1)
        exPhi, dPhi = _expand_vector(df_dec, "phi")
        if dPhi > 0:
            if phi_dim == 0:
                phi_dim = dPhi
            elif dPhi != phi_dim:
                raise ValueError(f"Inconsistent phi dimension across reps: expected {phi_dim}, got {dPhi}")
            part = pd.concat([part, exPhi], axis=1)

        exIt, dIt = _expand_vector(df_dec, "It")
        if dIt > 0:
            It_present = True
            part = pd.concat([part, exIt], axis=1)

        for k_on in ("attack_active", "onset_t", "attack_type"):
            if k_on in df_dec.columns and k_on not in part.columns:
                part[k_on] = df_dec[k_on].values

        rep_wm_dfs.append(part)
    dim_state = max(dims.get(x_key, 0), dims.get(y_key, 0), dims.get(yhat_key, 0))
    fig, axes = _square_fig(max(1, dim_state))
    t = None

    for i in range(max(1, dim_state)):
        ax = axes[i]
        cols = []
        for base_key in (x_key, yhat_key, y_key):
            cols.append(f"{base_key}_{i if dims[base_key] > 1 else 0}")

        wide = _align_reps(rep_scalar_dfs, time_col, cols)
        t = wide.index.to_numpy(dtype=float)

        m, lo, hi = _stats_from_wide(wide, cols[0], q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="x_true")

        m, lo, hi = _stats_from_wide(wide, cols[1], q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="y_hat")

        m, lo, hi = _stats_from_wide(wide, cols[2], q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="y")

        _vline_attack_onset(ax, df_all, time_col)
        ax.grid(True)
        ax.set_title(f"{label}: state/measurement dim {i}")

        if i == len(axes) - 1:
            ax.set_xlabel(time_col)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_1_state_measurement.png", dpi=200)
    plt.close(fig)

    dim_u = max(1, dims.get(u_key, 1))
    fig, axes = _square_fig(dim_u)

    for i in range(dim_u):
        ax = axes[i]
        col = f"{u_key}_{i if dims[u_key] > 1 else 0}"
        wide = _align_reps(rep_scalar_dfs, time_col, [col])
        t = wide.index.to_numpy(dtype=float)
        m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="u")
        _vline_attack_onset(ax, df_all, time_col)
        ax.grid(True)
        ax.set_title(f"{label}: control dim {i}")
        if i == dim_u - 1:
            ax.set_xlabel(time_col)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_2_control_u.png", dpi=200)
    plt.close(fig)

    dim_r = max(1, dims.get(r_key, 1))
    fig, axes = _square_fig(dim_r)

    for i in range(dim_r):
        ax = axes[i]
        col = f"{r_key}_{i if dims[r_key] > 1 else 0}"
        wide = _align_reps(rep_scalar_dfs, time_col, [col])
        t = wide.index.to_numpy(dtype=float)
        m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="r")
        _vline_attack_onset(ax, df_all, time_col)
        ax.grid(True)
        ax.set_title(f"{label}: residual dim {i}")
        if i == dim_r - 1:
            ax.set_xlabel(time_col)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_3_residual_r.png", dpi=200)
    plt.close(fig)

    dof = int(dims.get(y_key, 1))
    thr = float(chi2.ppf(1.0 - float(alpha), dof))

    fig, axes = _square_fig(1)
    ax = axes[0]
    col = f"{g_key}_0"
    wide = _align_reps(rep_scalar_dfs, time_col, [col])
    t = wide.index.to_numpy(dtype=float)
    m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
    _plot_band(ax, t, m, lo, hi, label="g")
    ax.axhline(thr, linestyle="--", label=f"threshold (alpha={alpha}, dof={dof})")
    _vline_attack_onset(ax, df_all, time_col)
    ax.grid(True)
    ax.set_title(f"{label}: test statistic")
    ax.set_xlabel(time_col)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_4_test_stat_g.png", dpi=200)
    plt.close(fig)

    fig, axes = _square_fig(1)
    ax = axes[0]
    col = f"{S1_key}_0"
    wide = _align_reps(rep_scalar_dfs, time_col, [col])
    t = wide.index.to_numpy(dtype=float)
    m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
    _plot_band(ax, t, m, lo, hi, label="S1")
    _vline_attack_onset(ax, df_all, time_col)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.set_title(f"{label}: belief")
    ax.set_xlabel(time_col)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_5_belief_S1.png", dpi=200)
    plt.close(fig)

    if phi_dim > 0:
        fig, axes = _square_fig(max(1, phi_dim))
        for i in range(max(1, phi_dim)):
            ax = axes[i]
            col = f"phi_{i if phi_dim > 1 else 0}"
            wide = _align_reps(rep_wm_dfs, time_col_dec, [col])
            t = wide.index.to_numpy(dtype=float)
            m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
            _plot_band(ax, t, m, lo, hi, label=f"phi[{i}]")
            _vline_attack_onset(ax, df_all_dec, time_col_dec)
            ax.grid(True)
            ax.set_title(f"{label}: watermark signal phi dim {i}")
            if i == phi_dim - 1:
                ax.set_xlabel(time_col_dec)
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_6_wm_signal_phi.png", dpi=200)
        plt.close(fig)

    if It_present:
        fig, axes = _square_fig(1)
        ax = axes[0]
        col = "It_0"
        wide = _align_reps(rep_wm_dfs, time_col_dec, [col])
        t = wide.index.to_numpy(dtype=float)
        m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
        _plot_band(ax, t, m, lo, hi, label="I_t")
        _vline_attack_onset(ax, df_all_dec, time_col_dec)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.set_title(f"{label}: alarms")
        ax.set_xlabel(time_col_dec)
        ax.legend(loc="best", fontsize=8)

        num_total, den_total = 0, 0
        for rep_df in rep_wm_dfs:
            n, d = _alarm_fraction_for_rep(rep_df, time_col=time_col_dec, label=label)
            num_total += int(n)
            den_total += int(d)
        if den_total > 0:
            ax.text(
                0.02,
                0.95,
                f"{num_total}/{den_total}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_7_alarms_It.png", dpi=200)
        plt.close(fig)

    U_plot_key = "U_applied" if U_dim_app > 0 else ("U" if U_dim_raw > 0 else None)
    U_plot_dim = U_dim_app if U_dim_app > 0 else U_dim_raw
    if U_plot_key is not None and U_plot_dim > 0:
        figsize = (max(6.0, 2.6 * U_plot_dim), max(5.0, 2.2 * U_plot_dim))
        fig, axes = plt.subplots(U_plot_dim, U_plot_dim, figsize=figsize, sharex=True)
        if U_plot_dim == 1:
            axes = np.asarray([[axes]])

        for i in range(U_plot_dim):
            for j in range(U_plot_dim):
                ax = axes[i, j]
                col = f"{U_plot_key}_{i}_{j}"
                wide = _align_reps(rep_wm_dfs, time_col_dec, [col])
                t = wide.index.to_numpy(dtype=float)
                m, lo, hi = _stats_from_wide(wide, col, q_lo, q_hi)
                _plot_band(ax, t, m, lo, hi, label=f"{U_plot_key}[{i},{j}]")
                _vline_attack_onset(ax, df_all_dec, time_col_dec)
                ax.grid(True)
                ax.set_title(f"{U_plot_key}[{i},{j}]", fontsize=9)
                if i == U_plot_dim - 1:
                    ax.set_xlabel(time_col_dec)

        title = f"{label}: applied watermark covariance" if U_plot_key == "U_applied" else f"{label}: applied watermark covariance"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"{label}_6_wm_cov_{U_plot_key}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir", type=str, required=True, help="Directory containing normal rep*/traj_*.jsonl")
    ap.add_argument("--attack_dir", type=str, required=True, help="Directory containing attack rep*/traj_*.jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write PNG figures")
    ap.add_argument("--q_lo", type=float, default=0.10)
    ap.add_argument("--q_hi", type=float, default=0.90)
    ap.add_argument("--alpha", type=float, default=None, help="Detector alpha (if not provided, tries run_summary.json else 0.01)")
    ap.add_argument("--prefer_processed", type=str, default="auto", choices=["auto", "processed", "decision"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plots_for_condition(
        run_dir=Path(args.normal_dir).expanduser(),
        label="normal",
        out_dir=out_dir,
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        alpha=args.alpha,
        prefer_processed=args.prefer_processed,
    )
    make_plots_for_condition(
        run_dir=Path(args.attack_dir).expanduser(),
        label="attack",
        out_dir=out_dir,
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        alpha=args.alpha,
        prefer_processed=args.prefer_processed,
    )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()

# python timeseries.py \
#   --normal_dir  output/simulation/<ENV>/<WM>/<normal_tag> \
#   --attack_dir  output/simulation/<ENV>/<WM>/<attack_tag> \
#   --out_dir     figs/<ENV> \
#   --q_lo 0.10 --q_hi 0.90

# python timeseries.py \
#     --normal_dir ./output/simulation/sm_dt_disc/const_cov_0.0/normal \
#     --attack_dir ./output/simulation/sm_dt_disc/const_cov_0.0/attack_replay_on20 \
#     --alpha 0.001 \
#     --out_dir output/simulation/sm_dt_disc/const_cov_0.0/img