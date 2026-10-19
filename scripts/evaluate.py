"""
Part D evaluation: score every trained model on its held-out sequence.

Loads the models written by scripts/train_ml.py and, for each target and
leave-one-sequence-out fold, predicts the windows of the sequence that fold
never trained or tuned on.

Writes, per target (the "experiment"):
  results/metrics/<target>.csv               MAE / RMSE / R^2 / bias per model per fold,
                                             plus a "mean" row (average over folds), and
                                             train_mae (same MAE on the training rows)
  results/metrics/<target>_importance.csv    RF impurity importance (train) and
                                             permutation importance (held-out) per fold
  results/metrics/<target>_failure_*.csv     failure analysis: opening-segment errors,
                                             train/test shift, error by rotation
                                             quartile, error concentration
  results/predictions/<target>.csv           one row per (fold, window, model)
  figures/ml/<target>_pred_vs_actual_<model>.png    one figure per model
  figures/ml/<target>_error_over_time_<seq>.png     one figure per held-out sequence
  figures/ml/<target>_residuals_<model>.png         one figure per model
  figures/ml/<target>_importance_rf.png
  figures/ml/<target>_importance_permutation.png

Usage:
    python scripts/evaluate.py                         # err_mag_m and err_mag_m_bc
    python scripts/evaluate.py --target err_mag_m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ml import DEFAULT_TARGETS, model_path
from vio_drift.evaluation import failure_analysis, plots
from vio_drift.evaluation.metrics import add_train_mae, mae, metrics_table
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import load_dataset, load_splits, split_xy, fold_frames
from vio_drift.models.ml.registry import MODEL_NAMES

PERMUTATION_MODELS = ("random_forest", "ridge")
PERMUTATION_REPEATS = 5


def predict_folds(df, folds, target, models_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Long held-out predictions table, training MAE per (model, fold), and the
    loaded models keyed by (fold_id, model).
    """
    rows, train_rows, fitted = [], [], {}
    for fold in folds:
        train, test = fold_frames(df, fold)
        X_train, y_train, _ = split_xy(train, target)
        X_test, y_test, _ = split_xy(test, target)
        for name in MODEL_NAMES:
            path = model_path(models_dir, target, fold.fold_id, name)
            if not path.exists():
                raise FileNotFoundError(f"{path.name} missing — run scripts/train_ml.py first")
            model = joblib.load(path)
            fitted[(fold.fold_id, name)] = model
            train_rows.append({"model": name, "fold_id": fold.fold_id,
                               "train_mae": mae(y_train, model.predict(X_train))})
            rows.append(pd.DataFrame({
                "fold_id": fold.fold_id,
                "seq_id": test["seq_id"],
                "window_id": test["window_id"],
                "t_start_ns": test["t_start_ns"],
                "model": name,
                "y_true": y_test,
                "y_pred": model.predict(X_test),
            }))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(train_rows), fitted


def importance_table(df, folds, target, fitted, seed) -> pd.DataFrame:
    """RF impurity importance and held-out permutation importance, per fold."""
    feats = feature_names()
    rows = []
    for fold in folds:
        _, test = fold_frames(df, fold)
        X_test, y_test, _ = split_xy(test, target)

        rf = fitted[(fold.fold_id, "random_forest")]
        rows.append(pd.DataFrame({
            "fold_id": fold.fold_id, "model": "random_forest", "kind": "impurity",
            "feature": feats, "importance": rf.feature_importances_, "importance_std": 0.0,
        }))
        for name in PERMUTATION_MODELS:
            perm = permutation_importance(
                fitted[(fold.fold_id, name)], X_test, y_test,
                scoring="neg_mean_absolute_error", n_repeats=PERMUTATION_REPEATS,
                random_state=seed, n_jobs=-1,
            )
            rows.append(pd.DataFrame({
                "fold_id": fold.fold_id, "model": name, "kind": "permutation",
                "feature": feats, "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }))
    return pd.concat(rows, ignore_index=True)


def save_figures(predictions, importance, target, fig_dir) -> None:
    figs = {}
    for model in predictions["model"].unique():
        figs[f"pred_vs_actual_{model}"] = plots.plot_pred_vs_actual(predictions, target, model)
        figs[f"residuals_{model}"] = plots.plot_residuals(predictions, target, model)
    for seq in predictions["seq_id"].unique():
        figs[f"error_over_time_{seq}"] = plots.plot_error_over_time(predictions, target, seq)

    rf = importance[(importance["model"] == "random_forest") & (importance["kind"] == "impurity")]
    rf_mean = rf.groupby("feature")["importance"].mean()
    figs["importance_rf"] = plots.plot_feature_importance(
        rf_mean, f"Random Forest impurity importance — {target}",
        "mean impurity importance over 3 folds")

    perm = importance[(importance["model"] == "random_forest") & (importance["kind"] == "permutation")]
    # Plotted in mm: the MAE increases are ~1e-4 m, unreadable as metre ticks.
    perm_mean = perm.groupby("feature")["importance"].mean() * 1000
    perm_std = perm.groupby("feature")["importance"].std() * 1000
    figs["importance_permutation"] = plots.plot_feature_importance(
        perm_mean, f"RF permutation importance (held-out) — {target}",
        "increase in held-out MAE when shuffled (mm), mean ± sd over folds",
        error=perm_std)

    for name, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_{name}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", action="append",
                        help="label column to evaluate (repeatable); default: both magnitudes")
    args = parser.parse_args()
    targets = args.target or DEFAULT_TARGETS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = int(config["seed"])

    models_dir = root / "models" / "ml"
    metrics_dir = root / "results" / "metrics"
    preds_dir = root / "results" / "predictions"
    fig_dir = root / "figures" / "ml"
    for d in (metrics_dir, preds_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    folds = load_splits(processed / "splits.json")

    for target in targets:
        print(f"\n{'=' * 60}\ntarget: {target}\n{'=' * 60}")
        df = load_dataset(processed, sequences, target=target)

        predictions, train_errors, fitted = predict_folds(df, folds, target, models_dir)
        predictions.to_csv(preds_dir / f"{target}.csv", index=False)

        table = add_train_mae(metrics_table(predictions), train_errors)
        table.to_csv(metrics_dir / f"{target}.csv", index=False)
        print(table.round(4).to_string(index=False))

        importance = importance_table(df, folds, target, fitted, seed)
        importance.to_csv(metrics_dir / f"{target}_importance.csv", index=False)

        failure = {
            "segments": failure_analysis.segment_errors(predictions),
            "shift": failure_analysis.feature_shift(df, folds, target),
            "rotation": failure_analysis.error_by_quartile(predictions, df),
            "spikes": failure_analysis.spike_stats(predictions),
        }
        for name, frame in failure.items():
            frame.to_csv(metrics_dir / f"{target}_failure_{name}.csv", index=False)

        save_figures(predictions, importance, target, fig_dir)
        print(f"[save] results/metrics/{target}.csv, results/predictions/{target}.csv, "
              f"figures/ml/{target}_*.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
