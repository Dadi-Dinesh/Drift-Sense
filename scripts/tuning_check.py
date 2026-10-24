"""
Diagnostic: which alpha does time-block CV pick, and is it a good one?

For each fold, model (Ridge, Lasso), number of blocks and picking rule
(median / best / 1-SE), tune alpha on the training flight only, refit on the whole
training flight, and score the held-out flight. Comparing these held-out
scores with scripts/alpha_sweep.py shows whether the CV choice lands near
the bottom of the true curve. Used once to choose the tuning setup, not
inside the final models.

Writes results/diagnostics/tuning_check_<target>.csv.

Usage:
    python scripts/tuning_check.py
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

TUNED = ("ridge", "lasso")
BLOCK_COUNTS = [3, 5, 8, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    df = load_dataset(processed, config["data"]["sequences"], target=args.target)
    features = feature_names()

    rows = []
    for fold in load_splits(processed / "splits.json"):
        train, test = fold_frames(df, fold)
        for model in TUNED:
            build = MODELS[model].build
            for n_blocks in BLOCK_COUNTS:
                result = tune_alpha(build, MODELS[model].alphas, train[features],
                                    train[args.target], n_blocks=n_blocks)
                for rule, alpha in [("median", result.median_alpha),
                                    ("best", result.best_alpha),
                                    ("1se", result.one_se_alpha)]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = build(alpha).fit(train[features], train[args.target])
                    s = score(test[args.target], fitted.predict(test[features]))
                    rows.append({"model": model, "test_seq": fold.test[0], "n_blocks": n_blocks,
                                 "rule": rule, "alpha": alpha, "test_mae": s["mae"], "test_r2": s["r2"]})

    out_df = pd.DataFrame(rows)
    out = root / "results" / "diagnostics" / f"tuning_check_{args.target}.csv"
    out_df.to_csv(out, index=False)
    for model in TUNED:
        t = out_df[out_df["model"] == model].pivot_table(
            index=["n_blocks", "rule"], columns="test_seq", values=["alpha", "test_mae", "test_r2"])
        print(f"\n{model}\n{t.to_string(float_format=lambda v: f'{v:.4g}')}")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
