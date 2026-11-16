"""
Trajectory Dead-Reckoning Drift Correction (R5 Technical Validation).

Subtracts predicted 3-axis drift errors [dx, dy, dz] from each window's
dead-reckoned end point to recover true vehicle position:
    p_corrected = p_dead_reckoned - pred_vector

Rebuilds continuous, non-overlapping trajectory paths from chained windows
to evaluate end-point error before vs after, and cumulative Absolute Trajectory
Error (ATE) across entire EuRoC flights.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def correct_endpoints(
    p_dead_reckoned: np.ndarray,
    pred_drift_vector: np.ndarray,
) -> np.ndarray:
    """
    Subtract predicted 3-axis drift displacement from dead-reckoned endpoints.
    
    Parameters
    ----------
    p_dead_reckoned : (N, 3) array of dead-reckoned endpoint positions (metres)
    pred_drift_vector : (N, 3) array of predicted [dx, dy, dz] drift errors (metres)
    
    Returns
    -------
    p_corrected : (N, 3) array of drift-corrected endpoint positions (metres)
    """
    p_dr = np.asarray(p_dead_reckoned, dtype=np.float64)
    p_pred = np.asarray(pred_drift_vector, dtype=np.float64)

    if p_dr.shape != p_pred.shape:
        raise ValueError(
            f"Shape mismatch: dead-reckoned {p_dr.shape} vs predicted {p_pred.shape}"
        )

    return p_dr - p_pred


def compute_endpoint_error_comparison(
    p_true: np.ndarray,
    p_raw: np.ndarray,
    pred_vector: np.ndarray,
) -> dict[str, Any]:
    """
    Compute endpoint position error statistics before and after correction.
    
    Returns dict with MAE, RMSE, max error, and percent reduction.
    """
    p_t = np.asarray(p_true, dtype=np.float64)
    p_r = np.asarray(p_raw, dtype=np.float64)
    p_c = correct_endpoints(p_r, pred_vector)

    err_before = np.linalg.norm(p_r - p_t, axis=1)
    err_after = np.linalg.norm(p_c - p_t, axis=1)

    mae_before = float(np.mean(err_before))
    mae_after = float(np.mean(err_after))
    rmse_before = float(np.sqrt(np.mean(err_before**2)))
    rmse_after = float(np.sqrt(np.mean(err_after**2)))
    max_before = float(np.max(err_before))
    max_after = float(np.max(err_after))

    reduction_pct = float((1.0 - (mae_after / mae_before)) * 100.0) if mae_before > 1e-12 else 0.0

    return {
        "mae_before": mae_before,
        "mae_after": mae_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "max_before": max_before,
        "max_after": max_after,
        "drift_reduction_pct": reduction_pct,
        "errors_before": err_before,
        "errors_after": err_after,
        "p_corrected": p_c,
    }


def select_non_overlapping_windows(windows_df: pd.DataFrame, step: int = 8) -> pd.DataFrame:
    """
    Select non-overlapping contiguous windows from sliding-window dataset.
    
    EuRoC windows: length 2.0 s (400 samples at 200 Hz), step 0.25 s (50 samples).
    stride = 2.0 s / 0.25 s = 8 windows for zero-overlap contiguous chaining.
    """
    df = windows_df.sort_values("t_start_ns").reset_index(drop=True)
    chain = df.iloc[::step].copy().reset_index(drop=True)
    return chain


def rebuild_trajectory(
    windows_chain: pd.DataFrame,
    aligned_df: pd.DataFrame,
    pred_vec: np.ndarray | None = None,
    p0: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Rebuild a continuous path from non-overlapping windows.
    
    Starting from p0 (initial true position):
      For window k:
        True segment displacement: Δp_true = p_true(t_end) - p_true(t_start)
        Raw dead-reckoned displacement: Δp_raw = Δp_true + err_vec
        Corrected displacement: Δp_corr = Δp_raw - pred_vec
      Cumulative position is integrated forward step-by-step.
    """
    chain = windows_chain.reset_index(drop=True)
    n_windows = len(chain)

    if pred_vec is None:
        pred_vec = np.zeros((n_windows, 3), dtype=np.float64)
    else:
        pred_vec = np.asarray(pred_vec, dtype=np.float64)
        if len(pred_vec) != n_windows:
            raise ValueError(f"pred_vec length {len(pred_vec)} does not match chain length {n_windows}")

    gt_map = aligned_df.set_index("t_ns")[["gt_px", "gt_py", "gt_pz"]]

    if p0 is None:
        t0 = chain.loc[0, "t_start_ns"]
        if t0 in gt_map.index:
            p0 = gt_map.loc[t0].to_numpy(dtype=np.float64)
        else:
            p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        p0 = np.asarray(p0, dtype=np.float64)

    true_path = [p0.copy()]
    raw_path = [p0.copy()]
    corr_path = [p0.copy()]

    rows_meta = [{
        "step": 0,
        "window_id": -1,
        "t_start_ns": chain.loc[0, "t_start_ns"],
        "t_end_ns": chain.loc[0, "t_start_ns"],
        "true_x": p0[0], "true_y": p0[1], "true_z": p0[2],
        "raw_x": p0[0], "raw_y": p0[1], "raw_z": p0[2],
        "corr_x": p0[0], "corr_y": p0[1], "corr_z": p0[2],
        "ate_raw": 0.0,
        "ate_corr": 0.0,
    }]

    curr_true = p0.copy()
    curr_raw = p0.copy()
    curr_corr = p0.copy()

    for i in range(n_windows):
        row = chain.iloc[i]
        t_start = row["t_start_ns"]
        t_end = row["t_end_ns"]

        pt_start = gt_map.loc[t_start].to_numpy(dtype=np.float64) if t_start in gt_map.index else curr_true
        pt_end = gt_map.loc[t_end].to_numpy(dtype=np.float64) if t_end in gt_map.index else curr_true

        delta_true = pt_end - pt_start
        err_vec = np.array([row["err_dx"], row["err_dy"], row["err_dz"]], dtype=np.float64)
        delta_raw = delta_true + err_vec

        pred = pred_vec[i]
        delta_corr = delta_raw - pred

        curr_true = curr_true + delta_true
        curr_raw = curr_raw + delta_raw
        curr_corr = curr_corr + delta_corr

        true_path.append(curr_true.copy())
        raw_path.append(curr_raw.copy())
        corr_path.append(curr_corr.copy())

        ate_raw = float(np.linalg.norm(curr_raw - curr_true))
        ate_corr = float(np.linalg.norm(curr_corr - curr_true))

        rows_meta.append({
            "step": i + 1,
            "window_id": int(row["window_id"]),
            "t_start_ns": t_start,
            "t_end_ns": t_end,
            "true_x": curr_true[0], "true_y": curr_true[1], "true_z": curr_true[2],
            "raw_x": curr_raw[0], "raw_y": curr_raw[1], "raw_z": curr_raw[2],
            "corr_x": curr_corr[0], "corr_y": curr_corr[1], "corr_z": curr_corr[2],
            "ate_raw": ate_raw,
            "ate_corr": ate_corr,
        })

    return pd.DataFrame(rows_meta)


def run_sequence_correction(
    seq_name: str,
    pred_vec: np.ndarray | pd.DataFrame | Path | str | None = None,
    processed_dir: Path | str | None = None,
    interim_dir: Path | str | None = None,
    predictions_dir: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Run full trajectory correction analysis on a flight sequence.
    
    If pred_vec is None:
      1. Tries results/predictions/E4vec_leave_<seq>.csv (learned residual 3-axis head)
      2. Tries data/processed/physics_leave_<seq>.csv (calibrated physics prior)
      3. Defaults to zero correction if neither is present.
    """
    root = Path(__file__).resolve().parents[3]
    processed_dir = Path(processed_dir) if processed_dir else root / "data" / "processed"
    interim_dir = Path(interim_dir) if interim_dir else root / "data" / "interim"
    predictions_dir = Path(predictions_dir) if predictions_dir else root / "results" / "predictions"

    w_df = pd.read_csv(processed_dir / f"windows_{seq_name}.csv")
    a_df = pd.read_csv(interim_dir / f"aligned_{seq_name}.csv")

    chain = select_non_overlapping_windows(w_df, step=8)

    # Resolve predicted drift vectors
    resolved_vec: np.ndarray | None = None

    if pred_vec is None:
        # Check E4vec prediction file
        e4vec_path = predictions_dir / f"E4vec_leave_{seq_name}.csv"
        if e4vec_path.exists():
            p_df = pd.read_csv(e4vec_path)
            merged = pd.merge(chain, p_df, on="window_id", suffixes=("", "_pred"))
            resolved_vec = merged[["y_pred_dx", "y_pred_dy", "y_pred_dz"]].to_numpy(dtype=np.float64)
        else:
            # Fall back to physics prior
            phys_path = processed_dir / f"physics_leave_{seq_name}.csv"
            if phys_path.exists():
                p_df = pd.read_csv(phys_path)
                merged = pd.merge(chain, p_df, on=["seq_id", "window_id"])
                resolved_vec = merged[["phys_dx", "phys_dy", "phys_dz"]].to_numpy(dtype=np.float64)
            else:
                resolved_vec = np.zeros((len(chain), 3), dtype=np.float64)

    elif isinstance(pred_vec, (str, Path)):
        p_df = pd.read_csv(pred_vec)
        merged = pd.merge(chain, p_df, on="window_id", suffixes=("", "_pred"))
        col_dx = "y_pred_dx" if "y_pred_dx" in merged.columns else "phys_dx"
        col_dy = "y_pred_dy" if "y_pred_dy" in merged.columns else "phys_dy"
        col_dz = "y_pred_dz" if "y_pred_dz" in merged.columns else "phys_dz"
        resolved_vec = merged[[col_dx, col_dy, col_dz]].to_numpy(dtype=np.float64)

    elif isinstance(pred_vec, pd.DataFrame):
        if "y_pred_dx" in pred_vec.columns and "window_id" in pred_vec.columns:
            merged = pd.merge(chain, pred_vec, on="window_id", suffixes=("", "_pred"))
            resolved_vec = merged[["y_pred_dx", "y_pred_dy", "y_pred_dz"]].to_numpy(dtype=np.float64)
        elif len(pred_vec) == len(chain):
            resolved_vec = pred_vec.to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"pred_vec DataFrame shape {pred_vec.shape} incompatible with chain length {len(chain)}")

    else:
        arr = np.asarray(pred_vec, dtype=np.float64)
        if len(arr) == len(chain):
            resolved_vec = arr
        elif len(arr) == len(w_df):
            # Full sequence array provided, slice by chain index
            resolved_vec = arr[chain.index]
        else:
            raise ValueError(f"pred_vec array length {len(arr)} does not match chain {len(chain)} or windows {len(w_df)}")

    gt_map = a_df.set_index("t_ns")[["gt_px", "gt_py", "gt_pz"]]
    p_end_true = []
    p_end_raw = []
    for _, r in chain.iterrows():
        pt = gt_map.loc[r["t_end_ns"]].to_numpy(dtype=np.float64)
        p_end_true.append(pt)
        p_end_raw.append(pt + np.array([r["err_dx"], r["err_dy"], r["err_dz"]], dtype=np.float64))

    p_end_true = np.array(p_end_true)
    p_end_raw = np.array(p_end_raw)

    comp = compute_endpoint_error_comparison(p_end_true, p_end_raw, resolved_vec)
    traj_df = rebuild_trajectory(chain, a_df, pred_vec=resolved_vec)

    comp["mean_ate_raw"] = float(traj_df["ate_raw"].mean())
    comp["final_ate_raw"] = float(traj_df["ate_raw"].iloc[-1])
    comp["mean_ate_corr"] = float(traj_df["ate_corr"].mean())
    comp["final_ate_corr"] = float(traj_df["ate_corr"].iloc[-1])
    comp["ate_reduction_pct"] = float((1.0 - (comp["mean_ate_corr"] / comp["mean_ate_raw"])) * 100.0)

    return comp, traj_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory drift correction on a sequence.")
    parser.add_argument("--seq", type=str, default="MH_01_easy", help="Sequence name (e.g. MH_01_easy)")
    parser.add_argument("--pred-csv", type=str, default=None, help="Path to prediction CSV with y_pred_dx, dy, dz")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output trajectory CSV path")
    args = parser.parse_args()

    print(f"\n--- Trajectory Correction: {args.seq} ---")
    comp, traj_df = run_sequence_correction(args.seq, pred_vec=args.pred_csv)
    print(f"Windows evaluated (non-overlapping): {len(traj_df) - 1}")
    print(f"Window Endpoint Error Before: {comp['mae_before']*100:.2f} cm (RMSE: {comp['rmse_before']*100:.2f} cm)")
    print(f"Window Endpoint Error After:  {comp['mae_after']*100:.2f} cm (RMSE: {comp['rmse_after']*100:.2f} cm)")
    print(f"Endpoint Error Reduction:     {comp['drift_reduction_pct']:.2f}%")
    print(f"Trajectory Final ATE Before:  {comp['final_ate_raw']:.2f} m")
    print(f"Trajectory Final ATE After:   {comp['final_ate_corr']:.2f} m")
    print(f"Mean ATE Reduction:           {comp['ate_reduction_pct']:.2f}%")

    if args.out_csv:
        out_p = Path(args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_csv(out_p, index=False)
        print(f"Trajectory saved to {out_p}")


if __name__ == "__main__":
    main()
