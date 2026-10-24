"""
Part D: fit every model on each fold's training flight, score the held-out one.

For each leave-one-sequence-out fold in data/processed/splits.json the model
is fit on the training flight only and scored on the test flight it never
saw. Models with a penalty (Ridge, Lasso) first choose alpha by time-block
CV on the training flight (median rule, 5 blocks — see
vio_drift.models.ml.tuning), then refit on the whole training flight.
Training MAE is reported too, so over-fitting shows up as a gap.

Writes:
  results/metrics/<target>.csv         one row per (fold, model) with the
                                       chosen alpha, plus a "mean" row per
                                       model averaging the two folds
  results/metrics/<target>_cv.csv      every alpha's CV MAE per block, per fold

Usage:
    python scripts/train_eval.py                    # target err_mag_m
    python scripts/train_eval.py --target err_dx
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.splits import load_splits
from vio_drift.evaluation.metrics import score
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import fold_frames, load_dataset
from vio_drift.models.ml.models import MODELS
from vio_drift.models.ml.tuning import tune_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m", help="label column to predict")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    df = load_dataset(processed, config["data"]["sequences"], target=args.target)
    features = feature_names()

    rows, cv_tables = [], []
    for fold in load_splits(processed / "splits.json"):
        train, test = fold_frames(df, fold)
        X, y = train[features], train[args.target]
        for name, spec in MODELS.items():
            alpha = None
            if spec.alphas is None:
                model = spec.build()
            else:
                tuned = tune_alpha(spec.build, spec.alphas, X, y)
                alpha = tuned.median_alpha
                model = spec.build(alpha)
                cv_tables.append(tuned.cv_table.assign(model=name, fold_id=fold.fold_id))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # Lasso convergence at tiny alpha
                model.fit(X, y)
            rows.append({
                "model": name,
                "fold_id": fold.fold_id,
                "test_seq": fold.test[0],
                "alpha": alpha,
                **score(test[args.target], model.predict(test[features])),
                "train_mae": score(y, model.predict(X))["mae"],
            })

    results = pd.DataFrame(rows)
    means = (results.drop(columns="alpha")
             .groupby("model", sort=False).mean(numeric_only=True).reset_index())
    results = pd.concat([results, means.assign(fold_id="mean", test_seq="both")], ignore_index=True)

    out = root / "results" / "metrics" / f"{args.target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    pd.concat(cv_tables, ignore_index=True).to_csv(out.with_name(f"{args.target}_cv.csv"), index=False)
    print(results.to_string(index=False, float_format="%.4f"))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
