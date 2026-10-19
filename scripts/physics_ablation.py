"""
Part D ablation: does the physics bias-error feature improve R^2?

Same leave-one-sequence-out folds and tuning as train_ml.py, run on three
feature sets:

  * base         — the 104 Part C window features (the main results);
  * physics      — only phys_dx, phys_dy, phys_dz, phys_mag
                   (features.physics; bias from the fold's training flights);
  * base+physics — both.

and on four targets: the headline magnitude ``err_mag_m`` and the 3-axis
error ``err_dx, err_dy, err_dz`` (world frame). The magnitude hides the
direction of the bias error, which turns with the drone's heading; the
axis targets keep it.

A fourth "model", ``physics_direct``, fits nothing: it predicts phys_mag
for the magnitude and phys_d<axis> for an axis. It shows how much of the
label the calibrated bias explains on its own.

Writes:
  results/ablations/physics.csv   MAE / RMSE / R^2 / bias per
                                  (target, features, model, flight) + "mean" rows

Usage:
    python scripts/physics_ablation.py
    python scripts/physics_ablation.py --target err_mag_m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import METRIC_COLS, regression_metrics
from vio_drift.features import physics
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import add_physics, fold_frames, load_dataset, load_splits
from vio_drift.models.ml.registry import model_specs
from vio_drift.models.ml.tuning import tune

TARGETS = ["err_mag_m", "err_dx", "err_dy", "err_dz"]
DIRECT_COLUMN = {"err_mag_m": "phys_mag", "err_dx": "phys_dx",
                 "err_dy": "phys_dy", "err_dz": "phys_dz"}


def feature_sets() -> dict[str, list[str]]:
    return {
        "base": feature_names(),
        "physics": physics.feature_names(),
        "base+physics": feature_names() + physics.feature_names(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", action="append", help="label column (repeatable)")
    args = parser.parse_args()
    targets = args.target or TARGETS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = int(config["seed"])
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = load_splits(processed / "splits.json")
    rows = []
    for target in targets:
        print(f"\n{'=' * 60}\ntarget: {target}\n{'=' * 60}")
        df = load_dataset(processed, sequences, target=target)
        for fold in folds:
            train, test = fold_frames(add_physics(df, processed, fold), fold)
            y_train, y_test = train[target].to_numpy(), test[target].to_numpy()
            groups = train["seq_id"].to_numpy()
            base = {"target": target, "test_seq": fold.test[0], "n": len(test)}

            rows.append({**base, "features": "physics", "model": "physics_direct",
                         **regression_metrics(y_test, test[DIRECT_COLUMN[target]].to_numpy())})

            for set_name, cols in feature_sets().items():
                for spec in model_specs(seed):
                    result = tune(spec.build(), spec.param_grid, train[cols], y_train, groups)
                    pred = result.best_estimator.predict(test[cols])
                    rows.append({**base, "features": set_name, "model": spec.name,
                                 **regression_metrics(y_test, pred)})
            print(f"[{fold.fold_id}] done")

    table = pd.DataFrame(rows)
    keys = ["target", "features", "model"]
    means = table.groupby(keys, as_index=False, sort=False)[["n", *METRIC_COLS]].mean()
    means["test_seq"] = "mean"
    table = pd.concat([table, means], ignore_index=True)
    out = out_dir / "physics.csv"
    table.to_csv(out, index=False)

    summary = means.pivot_table(index=["features", "model"], columns="target",
                                values="r2", sort=False)
    print("\nMean R^2 over the three held-out flights:")
    print(summary.round(3).to_string())
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
