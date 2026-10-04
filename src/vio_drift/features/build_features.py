"""
Combine every feature family into one row per window and hand off a table.

This is the glue for Part C. It turns a stack of raw windows
(shape ``(N, n_samples, 6)``, the ``windows_raw_<seq>.npy`` contract) into
a tidy table with one row per window and ~100 feature columns, then
attaches the drift label ``err_mag_m`` from Part A's ``windows_<seq>.csv``.

Column contract for the handoff file ``features_<seq>.csv``::

    seq_id, window_id, <feature columns...>, err_mag_m

The features are written **unscaled on purpose**. The project splits by
sequence (leave-one-sequence-out, see ``data/processed/splits.json``), so
any scaler must be fit on the training sequences of a fold only and then
applied to the held-out one — fitting a scaler across all sequences first
would leak test statistics into training. The scaler therefore belongs to
the modelling step, not to this file; :func:`fit_scaler` / :func:`apply_scaler`
are provided here so Part D can do exactly that, per fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import time_domain, frequency_domain, cross_axis

LABEL_COL = "err_mag_m"
ID_COLS = ("seq_id", "window_id")


def feature_names() -> list[str]:
    """All feature column names, in the order :func:`extract_window` emits them."""
    return (
        time_domain.feature_names()
        + frequency_domain.feature_names()
        + cross_axis.feature_names()
    )


def extract_window(window: np.ndarray, fs: float = 200.0) -> dict[str, float]:
    """Every feature for a single window of shape (n_samples, 6)."""
    row: dict[str, float] = {}
    row.update(time_domain.extract(window))
    row.update(frequency_domain.extract(window, fs=fs))
    row.update(cross_axis.extract(window, fs=fs))
    return row


def build_feature_frame(
    windows_raw: np.ndarray,
    seq_id: str,
    fs: float = 200.0,
) -> pd.DataFrame:
    """
    Feature table for one sequence's window stack (no label attached).

    Columns: ``seq_id, window_id, <features...>``. ``window_id`` is the row
    index into ``windows_raw``, matching Part A's convention.
    """
    if windows_raw.ndim != 3 or windows_raw.shape[2] != 6:
        raise ValueError(
            f"expected windows of shape (N, n_samples, 6), got {windows_raw.shape}"
        )

    rows = [extract_window(windows_raw[i], fs=fs) for i in range(windows_raw.shape[0])]
    frame = pd.DataFrame(rows, columns=feature_names())
    frame.insert(0, "window_id", np.arange(windows_raw.shape[0]))
    frame.insert(0, "seq_id", seq_id)
    return frame


def attach_label(features: pd.DataFrame, windows_csv: Path) -> pd.DataFrame:
    """
    Join the ``err_mag_m`` label from Part A's ``windows_<seq>.csv`` onto a
    feature frame by ``window_id``, returning the full handoff table in the
    contracted column order.
    """
    labels = pd.read_csv(windows_csv, usecols=["window_id", LABEL_COL])
    merged = features.merge(labels, on="window_id", how="left", validate="one_to_one")
    if merged[LABEL_COL].isna().any():
        missing = int(merged[LABEL_COL].isna().sum())
        raise ValueError(
            f"{missing} windows have no matching label in {windows_csv.name}; "
            "feature rows and label rows are out of sync."
        )
    ordered = [*ID_COLS, *feature_names(), LABEL_COL]
    return merged[ordered]


def build_sequence(
    seq_id: str,
    windows_raw_path: Path,
    windows_csv_path: Path,
    fs: float = 200.0,
) -> pd.DataFrame:
    """Load one sequence's raw windows, extract features, attach the label."""
    windows_raw = np.load(windows_raw_path)
    features = build_feature_frame(windows_raw, seq_id, fs=fs)
    return attach_label(features, windows_csv_path)


# ---------------------------------------------------------------------------
# Scaling helpers — for Part D to use per fold (fit on TRAIN sequences only).
# ---------------------------------------------------------------------------
def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """
    Fit a :class:`StandardScaler` on the feature columns of the training
    sequences only. Pass the concatenated feature rows of a fold's train
    sequences; never include the held-out test sequence.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names()].to_numpy())
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Return a copy of ``df`` with its feature columns standardised by an
    already-fitted ``scaler``. Id and label columns are left untouched.
    """
    out = df.copy()
    out[feature_names()] = scaler.transform(df[feature_names()].to_numpy())
    return out
