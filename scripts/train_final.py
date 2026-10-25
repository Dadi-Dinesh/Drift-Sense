"""
Train the final Lasso on both flights and save it as a .pkl file.

The held-out numbers in the report come from scripts/train_eval.py, where
each model is trained on one flight and scored on the other. This script
builds the model to *use* on new data: alpha is chosen by leave-one-flight-
out CV over both flights (vio_drift.models.ml.tuning.tune_alpha_by_flight),
then Lasso is refit on every window of both flights.

Writes:
  models/ml/lasso_<target>.pkl    the fitted sklearn Pipeline (scaler + Lasso)
  models/ml/lasso_<target>.json   what the model expects and how it was made

Load and predict:
    import pickle, pandas as pd
    model = pickle.load(open("models/ml/lasso_err_mag_m.pkl", "rb"))
    features = pd.read_csv("data/processed/features_MH_01_easy.csv")
    drift = model.predict(features[model.feature_names_in_])   # metres

Usage:
    python scripts/train_final.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import score
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import load_dataset
from vio_drift.models.ml.models import MODELS
from vio_drift.models.ml.tuning import tune_alpha_by_flight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", default="err_mag_m")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    df = load_dataset(processed, sequences, target=args.target)
    features = feature_names()
    X, y = df[features], df[args.target].to_numpy()

    spec = MODELS["lasso"]
    tuned = tune_alpha_by_flight(spec.build, spec.alphas, X, y, df["seq_id"].to_numpy())
    alpha = tuned.best_alpha
    print(tuned.cv_table.to_string(index=False, float_format="%.5f",
                                   formatters={"alpha": lambda a: f"{a:.3g}"}))
    print(f"\nchosen alpha = {alpha:.4g} (lowest mean leave-one-flight-out MAE)")

    model = spec.build(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    coef = model.named_steps["model"].coef_
    kept = [f for f, c in zip(features, coef) if c != 0]
    train = score(y, model.predict(X))

    out_dir = root / "models" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / f"lasso_{args.target}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(model, f)

    best = tuned.cv_table.loc[tuned.cv_table["alpha"] == alpha].iloc[0]
    meta = {
        "model": "Lasso (StandardScaler + sklearn.linear_model.Lasso)",
        "target": args.target,
        "target_units": "metres",
        "window_seconds": config["windowing"]["window_seconds"],
        "stride_seconds": config["windowing"]["stride_seconds"],
        "trained_on": sequences,
        "n_windows": int(len(df)),
        "alpha": alpha,
        "alpha_choice": "lowest mean leave-one-flight-out MAE over the Lasso grid in models.py",
        "cv_mae_by_held_out_flight": {k.removeprefix("flight_"): float(v)
                                      for k, v in best.items() if k.startswith("flight_")},
        "cv_note": ("alpha was chosen to minimise these CV scores, so they are slightly "
                    "optimistic; the unbiased held-out estimate is Lasso's mean MAE in "
                    "results/metrics/<target>.csv (scripts/train_eval.py)"),
        "train_mae": train["mae"],
        "train_r2": train["r2"],
        "n_features_in": len(features),
        "n_features_kept": len(kept),
        "features_kept": kept,
        "feature_order": features,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }
    (out_dir / f"lasso_{args.target}.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"kept {len(kept)} of {len(features)} features; train MAE {train['mae'] * 100:.2f} cm, "
          f"R² {train['r2']:.3f}")
    print(f"[save] {pkl}\n[save] {pkl.with_suffix('.json')}")


if __name__ == "__main__":
    main()
