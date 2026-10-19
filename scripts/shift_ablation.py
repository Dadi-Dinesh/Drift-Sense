"""
Part D ablation: can MH_03_medium be predicted better with only MH_01-03?

MH_03 fails because it is out of the training distribution (faster rotation,
wider and spikier label). Without new flights, we test four cheap changes
against the plain Ridge / Lasso models, each with the same nested
leave-one-sequence-out protocol as scripts/train_ml.py (hyperparameters tuned
on the training sequences only; the held-out sequence is scored once):

  * clip         — cap each feature at the 1st/99th percentile of the
                   training rows (learned inside the pipeline), so one extreme
                   window cannot dominate a prediction;
  * perflight    — standardise every flight by its *own* feature mean/std
                   (no labels used), removing flight-level offsets such as
                   MH_03's higher rotation rate;
  * context      — append, per feature, the mean over the current and previous
                   7 windows (2 s, causal, within the flight);
  * huber        — Huber loss (squared for small errors, linear for large),
                   so drift spikes pull the fit less.

Writes:
  results/ablations/<target>_shift.csv   MAE / RMSE / R^2 per (variant, model, fold),
                                         plus "mean" rows, and the chosen params

Usage:
    python scripts/shift_ablation.py                    # err_mag_m
    python scripts/shift_ablation.py --target err_mag_m_bc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import regression_metrics
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml import baselines
from vio_drift.models.ml.data import load_dataset, load_splits, fold_frames
from vio_drift.models.ml.tuning import tune

CONTEXT_WINDOWS = 8   # current + previous 7 windows = 2 s at 0.25 s stride


class PercentileClipper(TransformerMixin, BaseEstimator):
    """Clip each column to the [lower, upper] percentiles seen in ``fit``."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.low_ = np.percentile(X, self.lower, axis=0)
        self.high_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), self.low_, self.high_)


def with_clipper(pipe: Pipeline) -> Pipeline:
    """Insert a PercentileClipper before a pipeline's scaler."""
    return Pipeline([("clip", PercentileClipper()), *pipe.steps])


def build_huber() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(max_iter=2000)),
    ])


HUBER_GRID = {
    "huber__epsilon": [1.35, 2.0, 3.0],      # where squared loss turns linear (in robust std units)
    "huber__alpha": [1e-2, 1.0, 100.0],      # L2 strength
}


def model_variants() -> dict[str, tuple]:
    """name -> (builder, grid). Grids are the same as the main models'."""
    return {
        "ridge": (baselines.build_ridge, baselines.RIDGE_PARAM_GRID),
        "lasso": (baselines.build_lasso, baselines.LASSO_PARAM_GRID),
    }


def per_flight_standardise(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Standardise each flight's features by that flight's own mean and std."""
    out = df.copy()
    g = out.groupby("seq_id")[feats]
    out[feats] = (out[feats] - g.transform("mean")) / g.transform("std").replace(0, 1)
    return out


def add_context(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Append a causal rolling mean of every feature over CONTEXT_WINDOWS windows."""
    out = df.sort_values(["seq_id", "t_start_ns"]).copy()
    ctx = (out.groupby("seq_id")[feats]
              .transform(lambda s: s.rolling(CONTEXT_WINDOWS, min_periods=1).mean()))
    ctx.columns = [f"{c}__ctx2s" for c in feats]
    out = pd.concat([out, ctx], axis=1)
    return out, feats + list(ctx.columns)


def run_variant(df, feats, folds, target, variant, model_name, build, grid):
    rows = []
    for fold in folds:
        train, test = fold_frames(df, fold)
        res = tune(build(), grid, train[feats], train[target].to_numpy(),
                   train["seq_id"].to_numpy())
        pred = res.best_estimator.predict(test[feats])
        rows.append({
            "variant": variant, "model": model_name, "fold_id": fold.fold_id,
            "test_seq": fold.test[0],
            **regression_metrics(test[target].to_numpy(), pred),
            "best_params": json.dumps(res.best_params, default=str),
        })
        print(f"  {variant:<16s} {model_name:<6s} {fold.test[0]:<13s} "
              f"MAE={rows[-1]['mae']:.4f} R2={rows[-1]['r2']:+.3f}  {res.best_params}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m")
    args = parser.parse_args()
    target = args.target

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = load_splits(processed / "splits.json")
    df = load_dataset(processed, sequences, target=target)
    feats = feature_names()
    df_flight = per_flight_standardise(df, feats)
    df_ctx, feats_ctx = add_context(df, feats)

    # (variant, data, feature columns, clip?)
    setups = [
        ("baseline", df, feats, False),
        ("clip", df, feats, True),
        ("perflight", df_flight, feats, False),
        ("perflight+clip", df_flight, feats, True),
        ("context", df_ctx, feats_ctx, False),
    ]
    rows = []
    for variant, data, cols, clip in setups:
        for name, (build, grid) in model_variants().items():
            builder = (lambda b=build: with_clipper(b())) if clip else build
            g = {f"{k}": v for k, v in grid.items()}
            rows += run_variant(data, cols, folds, target, variant, name, builder, g)
    for variant, data in [("huber", df), ("perflight+huber", df_flight)]:
        rows += run_variant(data, feats, folds, target, variant, "huber", build_huber, HUBER_GRID)

    per_fold = pd.DataFrame(rows)
    metric_cols = ["mae", "rmse", "r2", "bias"]
    means = (per_fold.groupby(["variant", "model"], sort=False)
             .agg({"n": "sum", **{c: "mean" for c in metric_cols}})
             .reset_index().assign(fold_id="mean", test_seq="all", best_params=""))
    table = pd.concat([per_fold, means[per_fold.columns]], ignore_index=True)
    out = out_dir / f"{target}_shift.csv"
    table.to_csv(out, index=False)

    print("\nHeld-out R^2 by test flight (mean = average over folds):")
    print(table.pivot_table(index=["variant", "model"], columns="test_seq", values="r2",
                            sort=False).round(3).to_string())
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
