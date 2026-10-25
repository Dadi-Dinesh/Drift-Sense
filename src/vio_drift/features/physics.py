"""
Physics feature: the drift a *known* IMU bias would cause in one window.

Most of ``err_mag_m`` (~0.18 m of ~0.19 m per 1 s window, ~1.2 m per 2 s)
is the raw IMU's bias being integrated, and the EuRoC MH flights share one sensor whose
ground-truth biases are almost identical from flight to flight. So a bias
measured on the *training* flights is a fair calibration constant for a
new flight, the same way a real system reuses a previous calibration.

For each window the feature integrates the raw IMU twice from the same
starting state the label uses (``labels.drift_label``):

  * once as-is (exactly the label's prediction), and
  * once with the calibration bias ``b`` subtracted.

The difference ``d = p_raw - p_corrected`` is the position error that bias
alone accounts for. It is returned as world-frame components
``phys_dx, phys_dy, phys_dz`` and magnitude ``phys_mag``. With ``b`` equal
to the per-row ground-truth biases, ``d`` equals ``err_* - err_*_bc``
exactly (tests/test_physics_features.py).

``b`` must come from the training flights of a fold only
(:func:`training_bias`); taking it from the test flight would leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vio_drift.data.windowing import Window
from vio_drift.labels.imu_integration import (
    BIAS_ACCEL_COLS,
    BIAS_GYRO_COLS,
    integrate_window,
)

BIAS_COLS = BIAS_GYRO_COLS + BIAS_ACCEL_COLS


def feature_names() -> list[str]:
    return ["phys_dx", "phys_dy", "phys_dz", "phys_mag"]


def training_bias(aligned: dict[str, pd.DataFrame], train_seqs) -> np.ndarray:
    """
    Calibration bias ``[bgx, bgy, bgz, bax, bay, baz]``: the mean of each
    training flight's average ground-truth bias (flights weighted equally).
    """
    return np.mean([aligned[s][BIAS_COLS].mean().to_numpy() for s in train_seqs], axis=0)


def bias_error(
    window: Window,
    bias: np.ndarray,
    gravity_magnitude: float,
    gravity_sign: int,
    rotation_convention: str,
) -> np.ndarray:
    """World-frame position error ``d`` that ``bias`` causes over one window."""
    rows = window.rows.copy()
    p0 = rows.loc[0, ["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=np.float64)
    v0 = rows.loc[0, ["gt_vx", "gt_vy", "gt_vz"]].to_numpy(dtype=np.float64)
    q0 = rows.loc[0, ["gt_qw", "gt_qx", "gt_qy", "gt_qz"]].to_numpy(dtype=np.float64)
    conventions = dict(
        gravity_magnitude=gravity_magnitude,
        gravity_sign=gravity_sign,
        rotation_convention=rotation_convention,
    )

    raw = integrate_window(rows, p0, v0, q0, **conventions).p_end
    rows[BIAS_COLS] = np.broadcast_to(bias, (len(rows), len(BIAS_COLS)))
    corrected = integrate_window(rows, p0, v0, q0, subtract_bias=True, **conventions).p_end
    return raw - corrected


def physics_features(
    windows: list[Window],
    bias: np.ndarray,
    gravity_magnitude: float,
    gravity_sign: int,
    rotation_convention: str,
) -> pd.DataFrame:
    """One row per window: ``seq_id, window_id`` and :func:`feature_names`."""
    rows = []
    for w in windows:
        d = bias_error(w, bias, gravity_magnitude, gravity_sign, rotation_convention)
        rows.append({
            "seq_id": w.seq_name, "window_id": w.window_id,
            "phys_dx": d[0], "phys_dy": d[1], "phys_dz": d[2],
            "phys_mag": float(np.linalg.norm(d)),
        })
    return pd.DataFrame(rows)
