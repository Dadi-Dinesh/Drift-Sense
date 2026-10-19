"""
Part D ablation: does training on the start of a flight help on the rest of it?

MH_03_medium fails under leave-one-sequence-out because nothing like it is in
training. Here each flight in turn is split in time (data.chronological_split):
its first 60% of windows, minus a 4 s gap, may join training; its last 40% is
the test set. Every model is scored twice on the *same* test windows:

  * unseen   — trained on the other two flights only (as in train_ml.py);
  * adapted  — trained on the other two flights + the start of this flight;
  * mix60_40 — the adapted training set with easy windows randomly dropped
               until it is 60% easy / 40% medium by window count.

The difference is what seeing part of the flight is worth. Hyperparameters are
tuned with sequence-grouped CV on the training rows only; the flight's start
is its own group, so the inner loop also never mixes overlapping windows.

This answers a different question from the main results ("adapt once some of
the flight is seen", not "generalise to a new flight") and does not replace
them. A random row split is never used: overlapping windows would leak.

Writes:
  results/ablations/<target>_adapt.csv   MAE / RMSE / R^2 per (setting, model, flight),
                                         plus "mean" rows, and the chosen params

Usage:
    python scripts/adapt_ablation.py                    # err_mag_m
    python scripts/adapt_ablation.py --target err_mag_m_bc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import regression_metrics
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import chronological_split, load_dataset
from vio_drift.models.ml.registry import model_specs
from vio_drift.models.ml.tuning import tune

TRAIN_FRAC = 0.6   # first 60% of the held-out flight may be trained on
GAP_WINDOWS = 16   # 4 s at 0.25 s stride: no 2 s train window overlaps a test window
EASY_SHARE = 0.6   # mix60_40: fraction of training windows from easy flights


def rebalance(train: pd.DataFrame, easy_share: float, seed: int) -> pd.DataFrame:
    """Randomly drop easy windows so they make up ``easy_share`` of ``train``."""
    is_medium = train["seq_id"].str.contains("medium")
    medium, easy = train[is_medium], train[~is_medium]
    n_easy = round(len(medium) * easy_share / (1 - easy_share))
    if n_easy > len(easy):
        raise ValueError(f"only {len(easy)} easy windows, {n_easy} needed")
    return pd.concat([easy.sample(n_easy, random_state=seed), medium])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m")
    args = parser.parse_args()
    target = args.target

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    seed = config.get("seed", 42)
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(processed, sequences, target=target)
    feats = feature_names()

    rows = []
    for seq in sequences:
        others = df[df["seq_id"] != seq]
        start, test = chronological_split(df[df["seq_id"] == seq], TRAIN_FRAC, GAP_WINDOWS)
        start = start.assign(seq_id=f"{seq}_start")   # own group in the inner CV
        adapted = pd.concat([others, start])
        settings = {"unseen": others, "adapted": adapted,
                    "mix60_40": rebalance(adapted, EASY_SHARE, seed)}
        for setting, train in settings.items():
            for spec in model_specs(seed):
                res = tune(spec.build(), spec.param_grid, train[feats],
                           train[target].to_numpy(), train["seq_id"].to_numpy())
                pred = res.best_estimator.predict(test[feats])
                rows.append({
                    "setting": setting, "model": spec.name, "test_seq": seq,
                    "n_train": len(train),
                    **regression_metrics(test[target].to_numpy(), pred),
                    "best_params": json.dumps(res.best_params, default=str),
                })
                print(f"  {seq:<13s} {setting:<8s} {spec.name:<13s} "
                      f"MAE={rows[-1]['mae']:.4f} R2={rows[-1]['r2']:+.3f}")

    per_flight = pd.DataFrame(rows)
    metric_cols = ["mae", "rmse", "r2", "bias"]
    means = (per_flight.groupby(["setting", "model"], sort=False)
             .agg({"n": "sum", "n_train": "mean", **{c: "mean" for c in metric_cols}})
             .reset_index().assign(test_seq="mean", best_params=""))
    table = pd.concat([per_flight, means[per_flight.columns]], ignore_index=True)
    out = out_dir / f"{target}_adapt.csv"
    table.to_csv(out, index=False)

    print("\nR^2 on the last 40% of each flight (unseen / adapted / mix60_40):")
    print(table.pivot_table(index=["model"], columns=["test_seq", "setting"], values="r2",
                            sort=False).round(3).to_string())
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
