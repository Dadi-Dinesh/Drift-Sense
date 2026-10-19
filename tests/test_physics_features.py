"""
Checks for the bias-error physics feature (features/physics.py).

The toy tests need no data. The EuRoC test SKIPS unless the aligned
MH_01_easy table exists (run scripts/make_windows.py first).

Run with: pytest tests/test_physics_features.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.windowing import Window, make_windows
from vio_drift.features.physics import bias_error, training_bias
from vio_drift.labels.drift_label import compute_window_labels

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "interim" / "aligned_MH_01_easy.csv"


def _rest_window(accel_bias_x: float = 0.4, n: int = 200, rate_hz: int = 200) -> Window:
    dt_ns = int(1e9 / rate_hz)
    zeros = np.zeros(n)
    rows = pd.DataFrame({
        "t_ns": [i * dt_ns for i in range(n)],
        "gx": zeros, "gy": zeros, "gz": zeros,
        "ax": np.full(n, accel_bias_x), "ay": zeros, "az": np.full(n, 9.81),
        "gt_px": zeros, "gt_py": zeros, "gt_pz": zeros,
        "gt_vx": zeros, "gt_vy": zeros, "gt_vz": zeros,
        "gt_qw": np.ones(n), "gt_qx": zeros, "gt_qy": zeros, "gt_qz": zeros,
    })
    return Window(seq_name="toy", window_id=0, t_start_ns=0,
                  t_end_ns=int((n - 1) * dt_ns), rows=rows)


def test_zero_bias_gives_zero_feature():
    d = bias_error(_rest_window(), np.zeros(6), 9.81, -1, "body_to_world")
    assert np.allclose(d, 0.0)


def test_known_accel_bias_gives_half_b_t_squared():
    # 0.4 m/s^2 along body x for ~1 s at rest → about 0.5 * 0.4 * 1^2 = 0.2 m
    bias = np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0])
    d = bias_error(_rest_window(), bias, 9.81, -1, "body_to_world")
    assert d[0] == pytest.approx(0.2, abs=0.01)
    assert abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9


def test_training_bias_ignores_test_flight():
    a = pd.DataFrame({c: [1.0, 1.0] for c in
                      ["gt_bgx", "gt_bgy", "gt_bgz", "gt_bax", "gt_bay", "gt_baz"]})
    b = a * 3.0
    leak = a * 1000.0
    bias = training_bias({"A": a, "B": b, "TEST": leak}, ["A", "B"])
    assert np.allclose(bias, 2.0)


@pytest.mark.skipif(not ALIGNED.exists(), reason="aligned MH_01_easy not built")
def test_true_bias_reproduces_label_difference():
    """With the per-row GT biases, d must equal err_* - err_*_bc exactly."""
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["integration"]
    aligned = pd.read_csv(ALIGNED)
    windows = make_windows(aligned, "MH_01_easy", 1.0, 0.25, 200)[:5]
    for w in windows:
        lab = compute_window_labels(w, cfg["gravity_magnitude"], cfg["gravity_sign"],
                                    cfg["rotation_convention"])
        bias = w.rows[["gt_bgx", "gt_bgy", "gt_bgz", "gt_bax", "gt_bay", "gt_baz"]]
        # bias_error broadcasts one bias vector; the GT biases are constant
        # to ~1e-4 within a window, so use the window's own mean.
        d = bias_error(w, bias.mean().to_numpy(), cfg["gravity_magnitude"],
                       cfg["gravity_sign"], cfg["rotation_convention"])
        expected = np.array([lab.err_dx - lab.err_dx_bc,
                             lab.err_dy - lab.err_dy_bc,
                             lab.err_dz - lab.err_dz_bc])
        assert np.allclose(d, expected, atol=2e-4)
