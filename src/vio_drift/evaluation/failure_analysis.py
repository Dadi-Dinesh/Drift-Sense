"""
Failure analysis for Part D: *where* the held-out error lives and *why*.

Each function returns a small table that ``scripts/evaluate.py`` saves to
``results/metrics/<target>_failure_<name>.csv`` and notebook 06 displays, so
every failure-analysis number in the report traces back to a CSV.

Inputs are the long predictions table (one row per fold, window and model)
and the feature/label table from ``vio_drift.models.ml.data.load_dataset``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vio_drift.models.ml.data import Fold, fold_frames


def _with_time(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["t_s"] = (out["t_start_ns"] - out.groupby("seq_id")["t_start_ns"].transform("min")) / 1e9
    out["abs_err"] = (out["y_true"] - out["y_pred"]).abs()
    return out


def segment_errors(
    predictions: pd.DataFrame,
    split_s: float = 20.0,
    resume_s: float = 25.0,
) -> pd.DataFrame:
    """
    MAE per model in a sequence's opening segment (``t < split_s``) vs the
    rest (``t >= resume_s``); the transition between is left out. Also
    reports each segment's mean true label.
    """
    p = _with_time(predictions)
    p["segment"] = np.select(
        [p["t_s"] < split_s, p["t_s"] >= resume_s],
        [f"first_{split_s:g}s", f"after_{resume_s:g}s"],
        default="transition",
    )
    p = p[p["segment"] != "transition"]
    mae = p.pivot_table(index=["seq_id", "segment"], columns="model",
                        values="abs_err", aggfunc="mean")
    mae.columns = [f"mae_{c}" for c in mae.columns]
    first_model = p["model"].iloc[0]
    truth = (p[p["model"] == first_model]
             .groupby(["seq_id", "segment"])["y_true"].mean().rename("mean_y_true"))
    return pd.concat([truth, mae], axis=1).reset_index()


def feature_shift(
    df: pd.DataFrame,
    folds: list[Fold],
    target: str,
    features: tuple[str, ...] = ("gyro_mag_mean", "jerk_mag_mean", "ax_mean", "ax_rms"),
) -> pd.DataFrame:
    """
    For each fold, how far the held-out sequence sits from its training
    sequences: feature means, the share of test windows outside the training
    range, and the label's spread in train vs test.
    """
    rows = []
    for fold in folds:
        train, test = fold_frames(df, fold)
        for f in [*features, target]:
            outside = ((test[f] < train[f].min()) | (test[f] > train[f].max())).mean()
            rows.append({
                "fold_id": fold.fold_id, "test_seq": fold.test[0], "column": f,
                "train_mean": train[f].mean(), "test_mean": test[f].mean(),
                "train_std": train[f].std(), "test_std": test[f].std(),
                "test_max": test[f].max(),
                "test_outside_train_range_pct": 100.0 * outside,
            })
    return pd.DataFrame(rows)


def error_by_quartile(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    feature: str = "gyro_mag_mean",
    model: str = "random_forest",
) -> pd.DataFrame:
    """MAE of one model by within-sequence quartile of ``feature`` (Q1 = lowest)."""
    p = predictions[predictions["model"] == model].merge(
        df[["seq_id", "window_id", feature]], on=["seq_id", "window_id"])
    p["abs_err"] = (p["y_true"] - p["y_pred"]).abs()
    p["quartile"] = p.groupby("seq_id")[feature].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    out = p.pivot_table(index="seq_id", columns="quartile", values="abs_err",
                        aggfunc="mean", observed=True)
    out.columns = [f"mae_{c}" for c in out.columns]
    out["q4_over_q1"] = out["mae_Q4"] / out["mae_Q1"]
    return out.reset_index().assign(model=model, feature=feature)


def spike_stats(predictions: pd.DataFrame, model: str = "random_forest") -> pd.DataFrame:
    """
    How concentrated the error is, and whether it is a constant offset.

    * ``worst10_share`` — fraction of total absolute error in the worst 10% of windows.
    * ``corr_abs_err_vs_label_deviation`` — do big errors coincide with labels far from the median?
    * ``corr_pred_true`` — does the model track the label within the sequence?
    * ``bias2_share_of_mse`` — share of squared error explained by a constant offset.
    """
    rows = []
    for seq, g in predictions[predictions["model"] == model].groupby("seq_id", sort=False):
        resid = g["y_true"] - g["y_pred"]
        abs_err = resid.abs().sort_values(ascending=False)
        k = len(abs_err) // 10
        deviation = (g["y_true"] - g["y_true"].median()).abs()
        rows.append({
            "seq_id": seq, "model": model,
            "worst10_share": abs_err.head(k).sum() / abs_err.sum(),
            "corr_abs_err_vs_label_deviation": np.corrcoef(resid.abs(), deviation)[0, 1],
            "corr_pred_true": np.corrcoef(g["y_true"], g["y_pred"])[0, 1],
            "bias2_share_of_mse": resid.mean() ** 2 / (resid ** 2).mean(),
        })
    return pd.DataFrame(rows)
