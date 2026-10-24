"""
Load Part C's feature tables into the one table every Part D model uses.

One row per window: ``seq_id, window_id, t_start_ns, <104 features>, <target>``.
The target is joined from Part A's ``windows_<seq>.csv`` by ``window_id``, so
any label column there (``err_mag_m``, ``err_dx``, ``err_mag_m_bc`` ...) can
be predicted without rebuilding features. ``t_start_ns`` comes along so
predictions can be plotted over flight time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vio_drift.data.splits import Fold
from vio_drift.features.build_features import feature_names

ID_COLS = ["seq_id", "window_id", "t_start_ns"]


def load_dataset(processed_dir: Path, sequences: list[str], target: str = "err_mag_m") -> pd.DataFrame:
    """
    Concatenate each sequence's features with the chosen ``target``.

    Raises if features and labels do not describe the same windows.
    """
    processed_dir = Path(processed_dir)
    frames = []
    for seq in sequences:
        features = pd.read_csv(processed_dir / f"features_{seq}.csv")
        features = features.drop(columns=["err_mag_m"])
        labels = pd.read_csv(processed_dir / f"windows_{seq}.csv",
                             usecols=["window_id", "t_start_ns", target])
        merged = features.merge(labels, on="window_id", how="inner", validate="one_to_one")
        if len(merged) != len(features) or len(merged) != len(labels):
            raise ValueError(f"{seq}: feature and label windows do not match")
        frames.append(merged.sort_values("t_start_ns")[[*ID_COLS, *feature_names(), target]])
    return pd.concat(frames, ignore_index=True)


def fold_frames(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of a fold's training flights and of its held-out flight."""
    train = df[df["seq_id"].isin(fold.train)].reset_index(drop=True)
    test = df[df["seq_id"].isin(fold.test)].reset_index(drop=True)
    return train, test
