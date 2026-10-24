"""
Load Part C's feature tables and Part B's splits into the shape Part D needs.

Every model in Part D sees the same table: one row per window with
``seq_id, window_id, t_start_ns, <104 features>, <target>``. The target is
pulled from Part A's ``windows_<seq>.csv`` by ``window_id``, so any label
column there (``err_mag_m``, ``err_mag_m_bc``, ``err_dx`` ...) can be used
without regenerating features. ``t_start_ns`` comes along so predictions
can be plotted over flight time.

Splits are read from ``data/processed/splits.json`` (leave-one-sequence-out)
and re-checked here: a fold whose train and test sequences overlap raises
instead of silently leaking overlapping windows into the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vio_drift.features.build_features import feature_names

ID_COLS = ("seq_id", "window_id", "t_start_ns")

# 16 windows = 4 s at 0.25 s stride: twice the 2 s window length, so two
# windows this far apart share no IMU samples.
GAP_WINDOWS = 16


@dataclass(frozen=True)
class Fold:
    """One outer cross-validation fold: train on some sequences, test on one."""
    fold_id: str
    train: tuple[str, ...]
    test: tuple[str, ...]


def load_splits(splits_path: Path) -> list[Fold]:
    """Read leave-one-sequence-out folds from ``splits.json`` and validate them."""
    data = json.loads(Path(splits_path).read_text())
    folds = [
        Fold(fold_id=f["fold_id"], train=tuple(f["train"]), test=tuple(f["test"]))
        for f in data["folds"]
    ]
    for fold in folds:
        overlap = set(fold.train) & set(fold.test)
        if overlap:
            raise ValueError(f"{fold.fold_id}: {sorted(overlap)} in both train and test")
    return folds


def chronological_split(
    flight: pd.DataFrame,
    train_frac: float = 0.6,
    gap_windows: int = GAP_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one flight in time: the first ``train_frac`` of its windows go to
    train, the rest to test, and ``gap_windows`` windows before the cut are
    dropped so no training window overlaps a test window in time.

    The default gap of 16 windows (4 s at 0.25 s stride) is twice the 2 s
    window length, so the closest train and test windows share no samples.
    """
    flight = flight.sort_values("t_start_ns")
    cut = int(len(flight) * train_frac)
    if cut - gap_windows <= 0 or cut >= len(flight):
        raise ValueError(f"train_frac={train_frac} leaves an empty train or test part")
    return flight.iloc[: cut - gap_windows], flight.iloc[cut:]


def load_dataset(
    processed_dir: Path,
    sequences: list[str],
    target: str = "err_mag_m",
) -> pd.DataFrame:
    """
    Concatenate every sequence's feature table with the chosen ``target``.

    Raises if a feature row has no matching label row (or vice versa), since
    that would mean features and labels describe different windows.
    """
    processed_dir = Path(processed_dir)
    frames = []
    for seq in sequences:
        features = pd.read_csv(processed_dir / f"features_{seq}.csv")
        features = features.drop(columns=["err_mag_m"])
        labels = pd.read_csv(
            processed_dir / f"windows_{seq}.csv",
            usecols=["window_id", "t_start_ns", target],
        )
        if len(features) != len(labels):
            raise ValueError(
                f"{seq}: {len(features)} feature rows but {len(labels)} label rows"
            )
        merged = features.merge(labels, on="window_id", how="inner", validate="one_to_one")
        if len(merged) != len(features):
            raise ValueError(f"{seq}: window_id mismatch between features and labels")
        frames.append(merged[[*ID_COLS, *feature_names(), target]])
    return pd.concat(frames, ignore_index=True)


def split_xy(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(X, y, groups)``: feature matrix, target vector, sequence ids."""
    return df[feature_names()], df[target].to_numpy(), df["seq_id"].to_numpy()


def add_physics(df: pd.DataFrame, processed_dir: Path, fold: Fold) -> pd.DataFrame:
    """
    Join the fold's physics features (scripts/make_physics_features.py) onto
    ``df`` by window. They are per fold because the calibration bias comes
    from that fold's training flights only.
    """
    physics = pd.read_csv(Path(processed_dir) / f"physics_{fold.fold_id}.csv")
    merged = df.merge(physics, on=["seq_id", "window_id"], how="left", validate="one_to_one")
    if merged[physics.columns].isna().any().any():
        raise ValueError(f"{fold.fold_id}: windows missing from physics_{fold.fold_id}.csv")
    return merged


def fold_frames(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows for a fold's training sequences and for its held-out sequence."""
    train = df[df["seq_id"].isin(fold.train)].reset_index(drop=True)
    test = df[df["seq_id"].isin(fold.test)].reset_index(drop=True)
    return train, test
