"""
Diagnostic: Ridge and Lasso error against the penalty strength alpha.

For each fold and each alpha, fit on the training flight and record the
training and held-out MAE / R^2 (and, for Lasso, how many features keep a
non-zero weight). This shows the shape of the bias-variance curve. It uses
held-out scores, so it must NOT be used to pick alpha — that is tuning on
the test flight.

Writes:
  results/diagnostics/alpha_sweep_<target>.csv
  figures/ml/alpha_sweep_ridge.png, figures/ml/alpha_sweep_lasso.png

Usage:
    python scripts/alpha_sweep.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.splits import load_splits
from vio_drift.evaluation.metrics import score
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import fold_frames, load_dataset
from vio_drift.models.ml.models import build_lasso, build_ridge

ALPHAS = {
    "ridge": np.logspace(-3, 5, 17),
    "lasso": np.logspace(-6, -1, 16),
}
BUILDERS = {"ridge": build_ridge, "lasso": build_lasso}


def plot_sweep(sweep: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (seq, g) in enumerate(sweep[sweep["model"] == model].groupby("test_seq")):
        ax.plot(g["alpha"], g["mae"], marker="o", ms=3, color=f"C{i}",
                label=f"held-out MAE (test {seq})")
        ax.plot(g["alpha"], g["train_mae"], ls="--", color=f"C{i}",
                label=f"training MAE (train on the other flight)")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (penalty strength)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(f"{model.capitalize()}: error vs alpha")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        for model, alphas in ALPHAS.items():
            for alpha in alphas:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")    # Lasso convergence at tiny alpha
                    fitted = BUILDERS[model](alpha).fit(train[features], train[args.target])
                coef = fitted.named_steps["model"].coef_
                rows.append({
                    "model": model,
                    "alpha": alpha,
                    "test_seq": fold.test[0],
                    **score(test[args.target], fitted.predict(test[features])),
                    "train_mae": score(train[args.target], fitted.predict(train[features]))["mae"],
                    "n_nonzero": int(np.sum(coef != 0)),
                })

    sweep = pd.DataFrame(rows)
    out = root / "results" / "diagnostics" / f"alpha_sweep_{args.target}.csv"
    sweep.to_csv(out, index=False)

    fig_dir = root / "figures" / "ml"
    for model in ALPHAS:
        plot_sweep(sweep, model, fig_dir / f"alpha_sweep_{model}.png")

    for model in ALPHAS:
        table = sweep[sweep["model"] == model].pivot(index="alpha", columns="test_seq",
                                                      values=["mae", "r2", "train_mae", "n_nonzero"])
        print(f"\n{model}\n{table.to_string(float_format='%.4f')}")
    print(f"\n[save] {out}\n[save] {fig_dir}/alpha_sweep_{{ridge,lasso}}.png")


if __name__ == "__main__":
    main()
