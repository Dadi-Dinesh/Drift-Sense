"""
Cross-axis and vector-magnitude features for one window.

The per-axis features treat each channel on its own. But a drone moves as
one rigid body, so the axes are coupled: a banking turn lights up several
accelerometer and gyroscope axes together. Three kinds of feature capture
that coupling.

* **Vector magnitudes** ``|a|`` and ``|g|`` collapse the three accel / three
  gyro axes into one orientation-independent "amount of motion" signal, so
  the same manoeuvre looks the same regardless of heading.
* **Jerk** — the time-derivative of acceleration — flags abrupt changes.
  Sudden jerks are exactly when strapdown integration between samples is
  least accurate, so jerky windows are prime drift suspects.
* **Channel correlations** describe *how* the axes move together. Strong
  cross-axis correlation is the signature of coordinated motion (turns,
  climbs) as opposed to pure noise, which is uncorrelated.

Windows are (n_samples, 6) with the ``gx, gy, gz, ax, ay, az`` column
order, sampled at a fixed rate ``fs`` (200 Hz) used to scale the jerk.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .time_domain import AXES

_GYRO = (0, 1, 2)   # gx, gy, gz
_ACCEL = (3, 4, 5)  # ax, ay, az

# All unordered channel pairs, e.g. ("gx", "gy"), in a fixed order.
_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(AXES, 2))


def feature_names() -> list[str]:
    """Column names this module produces, in output order."""
    names = [
        "acc_mag_mean", "acc_mag_std", "acc_mag_max",
        "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max",
        "jerk_ax_rms", "jerk_ay_rms", "jerk_az_rms",
        "jerk_mag_mean", "jerk_mag_max",
    ]
    names += [f"corr_{a}_{b}" for a, b in _PAIRS]
    return names


def extract(window: np.ndarray, fs: float = 200.0) -> dict[str, float]:
    """
    Cross-axis features for one window of shape (n_samples, 6).

    Returns an ordered mapping whose keys match :func:`feature_names`.
    A pairwise correlation is 0.0 whenever either channel is constant
    (correlation is undefined there), so no NaNs reach the handoff table.
    """
    window = np.asarray(window, dtype=np.float64)
    out: dict[str, float] = {}

    accel = window[:, _ACCEL]
    gyro = window[:, _GYRO]

    acc_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    out["acc_mag_mean"] = float(acc_mag.mean())
    out["acc_mag_std"] = float(acc_mag.std())
    out["acc_mag_max"] = float(acc_mag.max())
    out["gyro_mag_mean"] = float(gyro_mag.mean())
    out["gyro_mag_std"] = float(gyro_mag.std())
    out["gyro_mag_max"] = float(gyro_mag.max())

    # Jerk = d(accel)/dt, per axis, scaled to m/s^3 by the sample spacing.
    jerk = np.diff(accel, axis=0) * fs
    if jerk.shape[0] == 0:
        jerk = np.zeros((1, 3))
    jerk_rms = np.sqrt(np.mean(jerk * jerk, axis=0))
    out["jerk_ax_rms"] = float(jerk_rms[0])
    out["jerk_ay_rms"] = float(jerk_rms[1])
    out["jerk_az_rms"] = float(jerk_rms[2])
    jerk_mag = np.linalg.norm(jerk, axis=1)
    out["jerk_mag_mean"] = float(jerk_mag.mean())
    out["jerk_mag_max"] = float(jerk_mag.max())

    # Pairwise Pearson correlation between every pair of channels.
    stds = window.std(axis=0)
    for a, b in _PAIRS:
        ia, ib = AXES.index(a), AXES.index(b)
        if stds[ia] == 0.0 or stds[ib] == 0.0:
            out[f"corr_{a}_{b}"] = 0.0
        else:
            out[f"corr_{a}_{b}"] = float(np.corrcoef(window[:, ia], window[:, ib])[0, 1])

    return out
