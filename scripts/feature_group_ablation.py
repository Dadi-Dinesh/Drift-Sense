"""
Part D ablation: which feature groups are actually needed?

Starting from all features (the 104 Part C features + the physics feature),
each group is removed in turn ("drop:<group>") and, separately, used on its
own ("only:<group>"). Same leave-one-sequence-out folds and grouped-CV tuning
as train_ml.py. A group is *necessary* if dropping it lowers held-out R^2;
it is *sufficient* if using it alone keeps most of the R^2.

Groups (by feature-name pattern):
  gyro_stats    gx_/gy_/gz_ time-domain statistics (mean, std, ..., zcr)
  accel_stats   ax_/ay_/az_ time-domain statistics
  frequency     *_dom_freq, *_spec_energy, *_spec_entropy
  magnitude     acc_mag_*, gyro_mag_*, jerk_*
  correlation   corr_*
  physics       phys_dx, phys_dy, phys_dz, phys_mag

Lasso (L1) also picks features by itself: its non-zero coefficients on the
all-features fit are saved per fold.

Writes:
  results/ablations/feature_groups.csv       R^2 / MAE per (target, model, setting, flight)
                                             + "mean" rows
  results/ablations/lasso_selected.csv       Lasso's non-zero standardised
                                             coefficients per (target, fold)

Usage:
    python scripts/feature_group_ablation.py
    python scripts/feature_group_ablation.py --target err_mag_m --model lasso
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
MODELS = ["lasso", "random_forest"]
FREQ_SUFFIXES = ("_dom_freq", "_spec_energy", "_spec_entropy")


def feature_groups() -> dict[str, list[str]]:
    base = feature_names()
    freq = [f for f in base if f.endswith(FREQ_SUFFIXES)]
    groups = {
        "gyro_stats": [f for f in base if f[:3] in ("gx_", "gy_", "gz_") and f not in freq],
        "accel_stats": [f for f in base if f[:3] in ("ax_", "ay_", "az_") and f not in freq],
        "frequency": freq,
        "magnitude": [f for f in base if f.startswith(("acc_mag_", "gyro_mag_", "jerk_"))],
        "correlation": [f for f in base if f.startswith("corr_")],
        "physics": physics.feature_names(),
    }
    grouped = sum(groups.values(), [])
    missing = set(base) - set(grouped)
    if missing or len(grouped) != len(set(grouped)):
        raise ValueError(f"feature groups must cover each feature once; unassigned: {missing}")
    return groups


def settings() -> dict[str, list[str]]:
    groups = feature_groups()
    every = sum(groups.values(), [])
    out = {"all": every}
    for name, cols in groups.items():
        out[f"drop:{name}"] = [f for f in every if f not in cols]
        out[f"only:{name}"] = cols
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", action="append", help="label column (repeatable)")
    parser.add_argument("--model", action="append", choices=MODELS, help="model (repeatable)")
    args = parser.parse_args()
    targets = args.target or TARGETS
    models = args.model or MODELS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = int(config["seed"])
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {s.name: s for s in model_specs(seed)}
    folds = load_splits(processed / "splits.json")
    rows, selected = [], []
    for target in targets:
        print(f"\n{'=' * 60}\ntarget: {target}\n{'=' * 60}")
        df = load_dataset(processed, sequences, target=target)
        for fold in folds:
            train, test = fold_frames(add_physics(df, processed, fold), fold)
            y_train, y_test = train[target].to_numpy(), test[target].to_numpy()
            groups = train["seq_id"].to_numpy()
            for setting, cols in settings().items():
                for name in models:
                    spec = specs[name]
                    result = tune(spec.build(), spec.param_grid, train[cols], y_train, groups)
                    pred = result.best_estimator.predict(test[cols])
                    rows.append({"target": target, "model": name, "setting": setting,
                                 "test_seq": fold.test[0], "n": len(test),
                                 "n_features": len(cols), **regression_metrics(y_test, pred)})
                    if name == "lasso" and setting == "all":
                        coef = result.best_estimator.named_steps["lasso"].coef_
                        selected += [{"target": target, "test_seq": fold.test[0],
                                      "feature": f, "coef_std": c}
                                     for f, c in zip(cols, coef) if c != 0]
            print(f"[{fold.fold_id}] done", flush=True)

    table = pd.DataFrame(rows)
    keys = ["target", "model", "setting"]
    means = table.groupby(keys, as_index=False, sort=False)[["n", "n_features", *METRIC_COLS]].mean()
    means["test_seq"] = "mean"
    pd.concat([table, means], ignore_index=True).to_csv(out_dir / "feature_groups.csv", index=False)
    pd.DataFrame(selected).to_csv(out_dir / "lasso_selected.csv", index=False)

    print("\nMean held-out R^2:")
    print(means.pivot_table(index="setting", columns=["model", "target"], values="r2",
                            sort=False).round(3).to_string())
    print(f"\n[save] {out_dir / 'feature_groups.csv'}, {out_dir / 'lasso_selected.csv'}")


if __name__ == "__main__":
    main()
