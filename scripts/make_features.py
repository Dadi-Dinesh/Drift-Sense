"""
Part C pipeline: raw windows -> feature table, for one or all sequences.

Reads Part A's handoff files from data/processed/:
  windows_raw_<seq>.npy   (N, 400, 6)  gx,gy,gz,ax,ay,az  (2.0 s windows)
  windows_<seq>.csv       for the err_mag_m label

Writes, per sequence:
  data/processed/features_<seq>.csv   seq_id, window_id, <features...>, err_mag_m

Features are written UNSCALED on purpose — scaling is fit on each fold's
training sequences only, at modelling time (see build_features.fit_scaler
and data/processed/splits.json). This script does not scale anything.

Usage:
    python scripts/make_features.py                 # all sequences in config
    python scripts/make_features.py MH_01_easy      # just one
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.features.build_features import build_sequence, feature_names


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())

    processed_dir = root / config["data"]["processed_dir"]
    fs = float(config["windowing"]["imu_rate_hz"])

    requested = sys.argv[1:]
    sequences = requested if requested else config["data"]["sequences"]

    n_features = len(feature_names())
    print(f"[make_features] {n_features} features per window, fs={fs:g} Hz")

    for seq_id in sequences:
        raw_path = processed_dir / f"windows_raw_{seq_id}.npy"
        csv_path = processed_dir / f"windows_{seq_id}.csv"
        if not raw_path.exists() or not csv_path.exists():
            print(f"[make_features] {seq_id}: missing Part A files, skipping "
                  f"({raw_path.name}, {csv_path.name})")
            continue

        table = build_sequence(seq_id, raw_path, csv_path, fs=fs)

        n_nan = int(table.isna().sum().sum())
        if n_nan:
            raise ValueError(f"{seq_id}: feature table has {n_nan} NaNs — refusing to write.")

        out_path = processed_dir / f"features_{seq_id}.csv"
        table.to_csv(out_path, index=False)
        print(
            f"[make_features] {seq_id}: wrote {out_path.name}  "
            f"({len(table)} windows x {table.shape[1]} cols; "
            f"err_mag_m mean={table['err_mag_m'].mean():.4f})"
        )

    print("\nDone. features_<seq>.csv are in data/processed/ — handoff to Part D.")


if __name__ == "__main__":
    main()
