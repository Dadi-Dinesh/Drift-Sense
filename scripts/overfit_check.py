"""
Part D over-fitting check: what tuning protects us from.

For each target and leave-one-sequence-out fold, compares two versions of
each tunable model on the same training rows:
  * tuned — the model saved by scripts/train_ml.py (best inner-CV setting),
  * loose — refit with the most flexible setting in that model's own grid
    (LOOSE_PARAMS below), i.e. the setting most able to memorise the
    training sequences.

A model that over-fits has a small train MAE and a much larger held-out
MAE, so ``ratio = test_mae / train_mae`` is the number to read.

Writes:
  results/metrics/<target>_overfit.csv   train/test MAE and ratio per
                                         (fold, model, setting), plus "mean" rows

Usage:
    python scripts/overfit_check.py                    # err_mag_m and err_mag_m_bc
    python scripts/overfit_check.py --target err_mag_m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ml import DEFAULT_TARGETS, model_path
from vio_drift.evaluation.metrics import mae
from vio_drift.models.ml.data import load_dataset, load_splits, split_xy, fold_frames
from vio_drift.models.ml.registry import model_specs

# Most flexible corner of each PARAM_GRID: weakest penalty, deepest trees.
LOOSE_PARAMS: dict[str, dict] = {
    "ridge": {"ridge__alpha": 0.01},
    "lasso": {"lasso__alpha": 1e-6},
    "random_forest": {"max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", action="append",
                        help="label column to check (repeatable); default: both magnitudes")
    args = parser.parse_args()
    targets = args.target or DEFAULT_TARGETS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = int(config["seed"])

    models_dir = root / "models" / "ml"
    metrics_dir = root / "results" / "metrics"
    folds = load_splits(processed / "splits.json")
    specs = {spec.name: spec for spec in model_specs(seed)}

    for target in targets:
        print(f"\n{'=' * 60}\ntarget: {target}\n{'=' * 60}")
        df = load_dataset(processed, sequences, target=target)
        rows = []

        for fold in folds:
            train, test = fold_frames(df, fold)
            X_train, y_train, _ = split_xy(train, target)
            X_test, y_test, _ = split_xy(test, target)

            for name, params in LOOSE_PARAMS.items():
                tuned = joblib.load(model_path(models_dir, target, fold.fold_id, name))
                loose = specs[name].build().set_params(**params).fit(X_train, y_train)
                for setting, model in (("tuned", tuned), ("loose", loose)):
                    rows.append({
                        "model": name, "setting": setting,
                        "fold_id": fold.fold_id, "test_seq": fold.test[0],
                        "train_mae": mae(y_train, model.predict(X_train)),
                        "test_mae": mae(y_test, model.predict(X_test)),
                    })

        per_fold = pd.DataFrame(rows)
        per_fold["ratio"] = per_fold["test_mae"] / per_fold["train_mae"]
        means = (
            per_fold.groupby(["model", "setting"], sort=False)[["train_mae", "test_mae", "ratio"]]
            .mean()
            .reset_index()
            .assign(fold_id="mean", test_seq="all")
        )
        table = pd.concat([per_fold, means[per_fold.columns]], ignore_index=True)

        out = metrics_dir / f"{target}_overfit.csv"
        table.to_csv(out, index=False)
        print(means.round(4).to_string(index=False))
        print(f"[save] {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
