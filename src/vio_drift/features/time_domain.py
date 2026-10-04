"""
Time-domain features, computed per IMU channel on one window.

A window is a (n_samples, 6) array whose columns are, in order,
``gx, gy, gz, ax, ay, az`` (gyroscope rad/s, then accelerometer m/s^2) —
the same column order Part A writes into ``windows_raw_<seq>.npy``.

Each statistic is a plain summary of one channel's 1-second slice. The
intuition for drift is that an open-loop IMU integrator accumulates error
fastest when the signal is energetic or erratic: large variance, big
peak-to-peak swings and frequent zero crossings all mean more motion to
integrate (and more sensor noise to accumulate), while a near-still,
low-variance window should drift very little. These hand-made statistics
are the classic-ML stand-in for "how hard was this window to integrate".
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Channel order of every window array in this project (see windowing.py).
AXES: tuple[str, ...] = ("gx", "gy", "gz", "ax", "ay", "az")

# Per-axis statistics, in the order they appear in the feature row.
_STATS: tuple[str, ...] = (
    "mean", "std", "var", "rms", "min", "max", "ptp", "skew", "kurt", "zcr",
)


def feature_names() -> list[str]:
    """Column names this module produces, in output order."""
    return [f"{axis}_{stat}" for axis in AXES for stat in _STATS]


def _zero_crossing_rate(x: np.ndarray) -> float:
    """
    Fraction of adjacent sample pairs that cross the signal's own mean.

    We cross about the per-window mean, not zero: the accelerometer axes
    carry a large gravity/DC offset, so a raw-zero crossing rate would
    mostly report which way gravity points rather than how the signal
    oscillates. Removing the mean first makes the rate a measure of
    oscillation for every channel alike.
    """
    if x.size < 2:
        return 0.0
    centered = x - x.mean()
    signs = np.signbit(centered)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (x.size - 1))


def extract(window: np.ndarray) -> dict[str, float]:
    """
    Time-domain features for one window of shape (n_samples, 6).

    Returns an ordered mapping ``{"<axis>_<stat>": value}`` whose keys
    match :func:`feature_names`. All values are finite: for a constant
    channel (zero variance) skewness and kurtosis are reported as 0.0
    rather than NaN, so the handoff table never carries NaNs.
    """
    window = np.asarray(window, dtype=np.float64)
    out: dict[str, float] = {}

    for i, axis in enumerate(AXES):
        x = window[:, i]
        mean = float(x.mean())
        std = float(x.std())
        out[f"{axis}_mean"] = mean
        out[f"{axis}_std"] = std
        out[f"{axis}_var"] = float(x.var())
        out[f"{axis}_rms"] = float(np.sqrt(np.mean(x * x)))
        out[f"{axis}_min"] = float(x.min())
        out[f"{axis}_max"] = float(x.max())
        out[f"{axis}_ptp"] = float(np.ptp(x))
        # skew/kurtosis are undefined for a flat signal; report 0.0 there.
        if std == 0.0:
            out[f"{axis}_skew"] = 0.0
            out[f"{axis}_kurt"] = 0.0
        else:
            out[f"{axis}_skew"] = float(stats.skew(x))
            out[f"{axis}_kurt"] = float(stats.kurtosis(x))  # Fisher (excess)
        out[f"{axis}_zcr"] = _zero_crossing_rate(x)

    return out
