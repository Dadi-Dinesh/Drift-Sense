"""
Regression metrics for drift prediction, written out so each is easy to
explain.

For true labels ``y`` and predictions ``p`` over ``n`` windows:

  * **MAE**  = mean(|y - p|) — average miss, in metres.
  * **RMSE** = sqrt(mean((y - p)^2)) — like MAE but punishes big misses more.
  * **R^2**  = 1 - sum((y - p)^2) / sum((y - mean(y))^2) — fraction of the
    test sequence's own label variance explained. 0 means "no better than
    knowing the test mean"; negative means worse than that. Note a
    train-mean baseline usually scores *below* 0, because the train mean is
    not the test mean.
  * **bias** = mean(p - y) — systematic over (+) or under (-) prediction.
    Not requested in the brief but useful for failure analysis: it shows
    how much of the error is a constant offset between sequences.

``train_mae`` (added by :func:`add_train_mae`) is the same MAE on the rows
the model was fitted on. Comparing it with the held-out ``mae`` shows
over-fitting: a model that memorises its training sequences has a tiny
``train_mae`` and a much larger held-out ``mae``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLS = ("mae", "rmse", "r2", "bias")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff * diff)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the mean of ``y_true``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ss_res = float(np.sum((y_true - np.asarray(y_pred)) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error, prediction minus truth."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All metrics for one set of predictions."""
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fold metrics plus a ``mean`` row (average over folds) for each model.

    ``predictions`` is the long table written by ``scripts/evaluate.py``:
    one row per (fold, window, model) with ``y_true`` and ``y_pred``.
    """
    rows = []
    for (model, fold_id, seq_id), g in predictions.groupby(
        ["model", "fold_id", "seq_id"], sort=False
    ):
        rows.append({
            "model": model, "fold_id": fold_id, "test_seq": seq_id,
            **regression_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
        })
    per_fold = pd.DataFrame(rows)

    means = (
        per_fold.groupby("model", sort=False)[["n", *METRIC_COLS]]
        .agg({"n": "sum", **{c: "mean" for c in METRIC_COLS}})
        .reset_index()
        .assign(fold_id="mean", test_seq="all")
    )
    return pd.concat([per_fold, means[per_fold.columns]], ignore_index=True)


def add_train_mae(table: pd.DataFrame, train_errors: pd.DataFrame) -> pd.DataFrame:
    """
    Append a ``train_mae`` column to a :func:`metrics_table` result.

    ``train_errors`` has one row per (model, fold_id) with ``train_mae``;
    each ``mean`` row gets that model's average over folds.
    """
    per_fold = table[table["fold_id"] != "mean"].merge(
        train_errors[["model", "fold_id", "train_mae"]], on=["model", "fold_id"], how="left",
    )
    means = table[table["fold_id"] == "mean"].merge(
        per_fold.groupby("model", as_index=False)["train_mae"].mean(), on="model", how="left",
    )
    return pd.concat([per_fold, means], ignore_index=True)
