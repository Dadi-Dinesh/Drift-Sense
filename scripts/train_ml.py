"""
Part D training: tune and fit every model on every leave-one-sequence-out fold.

For each target and each outer fold in data/processed/splits.json:
  1. take the fold's training sequences only,
  2. grid-search each model's hyperparameters with sequence-grouped inner CV
     (vio_drift.models.ml.tuning — the held-out sequence is never seen),
  3. refit the best setting on all training rows and save it.

Writes:
  models/ml/<target>__<fold_id>__<model>.joblib   fitted models (git-ignored)
  results/metrics/<target>_tuning.csv             best params + inner-CV MAE

Scoring the held-out sequence is scripts/evaluate.py's job.

Usage:
    python scripts/train_ml.py                           # err_mag_m and err_mag_m_bc
    python scripts/train_ml.py --target err_mag_m
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.models.ml.data import load_dataset, load_splits, split_xy, fold_frames
from vio_drift.models.ml.registry import model_specs
from vio_drift.models.ml.tuning import tune

DEFAULT_TARGETS = ["err_mag_m", "err_mag_m_bc"]


def model_path(models_dir: Path, target: str, fold_id: str, model: str) -> Path:
    """Where a fitted model for (target, fold, model) is saved."""
    return models_dir / f"{target}__{fold_id}__{model}.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", action="append",
                        help="label column to predict (repeatable); default: both magnitudes")
    args = parser.parse_args()
    targets = args.target or DEFAULT_TARGETS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = int(config["seed"])

    models_dir = root / "models" / "ml"
    metrics_dir = root / "results" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    folds = load_splits(processed / "splits.json")

    for target in targets:
        print(f"\n{'=' * 60}\ntarget: {target}\n{'=' * 60}")
        df = load_dataset(processed, sequences, target=target)
        rows = []

        for fold in folds:
            train, _ = fold_frames(df, fold)
            X, y, groups = split_xy(train, target)
            print(f"\n[{fold.fold_id}] train={list(fold.train)} ({len(train)} windows)")

            for spec in model_specs(seed):
                t0 = time.perf_counter()
                result = tune(spec.build(), spec.param_grid, X, y, groups)
                seconds = time.perf_counter() - t0

                joblib.dump(result.best_estimator,
                            model_path(models_dir, target, fold.fold_id, spec.name))
                rows.append({
                    "target": target,
                    "fold_id": fold.fold_id,
                    "test_seq": fold.test[0],
                    "model": spec.name,
                    "inner_cv_mae": result.inner_cv_mae,
                    "n_candidates": len(result.cv_results),
                    "best_params": json.dumps(result.best_params, default=str),
                    "fit_seconds": round(seconds, 1),
                })
                print(f"  {spec.name:<14s} inner-CV MAE={result.inner_cv_mae:.4f}  "
                      f"{len(result.cv_results):>2d} candidates  {seconds:5.1f}s  "
                      f"best={result.best_params}")

        out = metrics_dir / f"{target}_tuning.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[save] {out}")

    print("\nDone. Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
