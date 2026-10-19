"""
Diagnostic: would a longer window give a more predictable label?

The 1 s magnitude label is ~0.18 m of bias-driven drift plus a 2-3 cm
residual that is white noise from window to window (it is what is left
after the ground-truth bias is removed, ``err_*_bc``). Bias drift grows
roughly with t^2-t^3 and the residual more slowly, so a longer window may
raise the share of the label that can be predicted.

For each window length, every flight is re-windowed and re-labelled
(drift_label, unchanged), and on each leave-one-sequence-out test flight:

  * ``physics_direct`` — R^2 of the physics feature with no fitting
    (phys_mag for the magnitude, phys_d<axis> for an axis), bias from the
    fold's training flights;
  * ``physics_calibrated`` — magnitude only: y ~ a + c * phys_mag fitted on
    the training flights;
  * ``oracle_ceiling`` — magnitude only: in-sample R^2 of y ~ a + c * |d_true|
    on the test flight, where d_true = err - err_bc uses the test flight's
    own GT bias. No real model can know d_true, so this bounds what a
    bias-based predictor of the magnitude can reach.

Features for other window lengths are not rebuilt; this checks the label,
not the 104-feature models. It does not change the 1 s hand-off.

Writes results/diagnostics/window_length_ceiling.csv

Usage:
    python scripts/window_length_ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.windowing import make_windows
from vio_drift.evaluation.metrics import r2
from vio_drift.features.physics import bias_error, training_bias
from vio_drift.labels.drift_label import compute_window_labels
from vio_drift.models.ml.data import load_splits

WINDOW_LENGTHS = [0.5, 1.0, 1.5, 2.0]
STRIDE_SECONDS = 0.5  # coarser than the 0.25 s hand-off to keep this quick
AXES = ("dx", "dy", "dz")


def label_table(windows, biases, integ) -> pd.DataFrame:
    """Labels, d_true and the physics feature under each fold's bias."""
    conv = (integ["gravity_magnitude"], integ["gravity_sign"], integ["rotation_convention"])
    rows = []
    for w in windows:
        lab = compute_window_labels(w, *conv)
        if lab is None:
            continue
        row = {"err_mag_m": lab.err_mag_m}
        d_true = []
        for ax in AXES:
            row[f"err_{ax}"] = getattr(lab, f"err_{ax}")
            d_true.append(row[f"err_{ax}"] - getattr(lab, f"err_{ax}_bc"))
        row["d_true_mag"] = float(np.linalg.norm(d_true))
        for fold_id, bias in biases.items():
            d = bias_error(w, bias, *conv)
            row.update({f"{fold_id}:phys_{ax}": d[k] for k, ax in enumerate(AXES)})
            row[f"{fold_id}:phys_mag"] = float(np.linalg.norm(d))
        rows.append(row)
    return pd.DataFrame(rows)


def linear_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    interim = root / config["data"]["interim_dir"]
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    rate = config["windowing"]["imu_rate_hz"]
    integ = config["integration"]

    folds = load_splits(processed / "splits.json")
    aligned = {s: pd.read_csv(interim / f"aligned_{s}.csv") for s in sequences}
    biases = {f.fold_id: training_bias(aligned, f.train) for f in folds}

    rows = []
    for length in WINDOW_LENGTHS:
        tables = {}
        for s in sequences:
            windows = make_windows(aligned[s], s, length, STRIDE_SECONDS, int(length * rate))
            tables[s] = label_table(windows, biases, integ)

        for fold in folds:
            test = tables[fold.test[0]]
            train = pd.concat([tables[s] for s in fold.train], ignore_index=True)
            f = fold.fold_id
            y = test["err_mag_m"].to_numpy()
            base = {"window_seconds": length, "test_seq": fold.test[0], "n": len(test),
                    "label_mean": y.mean(), "label_std": y.std()}

            a, c = linear_fit(train[f"{f}:phys_mag"].to_numpy(), train["err_mag_m"].to_numpy())
            oa, oc = linear_fit(test["d_true_mag"].to_numpy(), y)
            rows += [
                {**base, "target": "err_mag_m", "method": "physics_direct",
                 "r2": r2(y, test[f"{f}:phys_mag"].to_numpy())},
                {**base, "target": "err_mag_m", "method": "physics_calibrated",
                 "r2": r2(y, a + c * test[f"{f}:phys_mag"].to_numpy())},
                {**base, "target": "err_mag_m", "method": "oracle_ceiling",
                 "r2": r2(y, oa + oc * test["d_true_mag"].to_numpy())},
            ]
            for ax in AXES:
                y_ax = test[f"err_{ax}"].to_numpy()
                rows.append({**base, "target": f"err_{ax}", "method": "physics_direct",
                             "label_mean": y_ax.mean(), "label_std": y_ax.std(),
                             "r2": r2(y_ax, test[f"{f}:phys_{ax}"].to_numpy())})
        print(f"[window {length:g} s] done")

    table = pd.DataFrame(rows)
    means = table.groupby(["window_seconds", "target", "method"], as_index=False, sort=False)[
        ["n", "label_mean", "label_std", "r2"]].mean()
    means["test_seq"] = "mean"
    out = root / "results" / "diagnostics" / "window_length_ceiling.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table, means], ignore_index=True).to_csv(out, index=False)

    print("\nMean held-out R^2 by window length:")
    print(means.pivot_table(index=["target", "method"], columns="window_seconds",
                            values="r2", sort=False).round(3).to_string())
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
