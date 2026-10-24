"""
Physics feature tables, one per leave-one-sequence-out fold.

The calibration bias is the mean ground-truth IMU bias of the fold's
*training* flights only (features.physics.training_bias), so each fold gets
its own table: the held-out flight's bias is never used, not even for its
own windows.

Reads:
  data/interim/aligned_<seq>.csv      raw IMU + ground-truth start states
  data/processed/windows_<seq>.csv    which window_ids survived labelling
  data/processed/splits.json          the folds

Writes:
  data/processed/physics_<fold_id>.csv   seq_id, window_id, phys_dx, phys_dy,
                                         phys_dz, phys_mag (every sequence)

Usage:
    python scripts/make_physics_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.windowing import make_windows
from vio_drift.features.physics import physics_features, training_bias
from vio_drift.data.splits import load_splits


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    interim = root / config["data"]["interim_dir"]
    processed = root / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    win = config["windowing"]
    integ = config["integration"]
    expected_rows = int(win["window_seconds"] * win["imu_rate_hz"])

    aligned = {s: pd.read_csv(interim / f"aligned_{s}.csv") for s in sequences}
    windows = {}
    for s in sequences:
        kept = set(pd.read_csv(processed / f"windows_{s}.csv", usecols=["window_id"])["window_id"])
        all_windows = make_windows(aligned[s], s, win["window_seconds"],
                                   win["stride_seconds"], expected_rows)
        windows[s] = [w for w in all_windows if w.window_id in kept]

    for fold in load_splits(processed / "splits.json"):
        bias = training_bias(aligned, fold.train)
        print(f"[{fold.fold_id}] bias from {list(fold.train)}: "
              f"gyro={bias[:3].round(4)} accel={bias[3:].round(4)}")
        table = pd.concat(
            [physics_features(windows[s], bias, integ["gravity_magnitude"],
                              integ["gravity_sign"], integ["rotation_convention"])
             for s in sequences],
            ignore_index=True,
        )
        out = processed / f"physics_{fold.fold_id}.csv"
        table.to_csv(out, index=False)
        print(f"  wrote {out.name}: {len(table)} windows, "
              f"phys_mag mean={table['phys_mag'].mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
