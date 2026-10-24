"""
Part D: fit every model on each fold's training flight, score the held-out one.

For each leave-one-sequence-out fold in data/processed/splits.json the model
is fit on the training flight only and scored on the test flight it never
saw. Training MAE is reported too, so over-fitting shows up as a gap.

Writes results/metrics/<target>.csv: one row per (fold, model), plus a
"mean" row per model averaging the two folds.

Usage:
    python scripts/train_eval.py                    # target err_mag_m
    python scripts/train_eval.py --target err_dx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.splits import load_splits
from vio_drift.evaluation.metrics import score
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import fold_frames, load_dataset
from vio_drift.models.ml.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m", help="label column to predict")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    df = load_dataset(processed, config["data"]["sequences"], target=args.target)
    features = feature_names()

    rows = []
    for fold in load_splits(processed / "splits.json"):
        train, test = fold_frames(df, fold)
        for name, build in MODELS.items():
            model = build().fit(train[features], train[args.target])
            rows.append({
                "model": name,
                "fold_id": fold.fold_id,
                "test_seq": fold.test[0],
                **score(test[args.target], model.predict(test[features])),
                "train_mae": score(train[args.target], model.predict(train[features]))["mae"],
            })

    results = pd.DataFrame(rows)
    means = results.groupby("model", sort=False).mean(numeric_only=True).reset_index()
    results = pd.concat([results, means.assign(fold_id="mean", test_seq="both")], ignore_index=True)

    out = root / "results" / "metrics" / f"{args.target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(results.to_string(index=False, float_format="%.4f"))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
