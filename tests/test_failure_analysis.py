"""
Toy-data checks for the Part D failure-analysis tables.

Run with: pytest tests/test_failure_analysis.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation import failure_analysis as fa


def _preds(n: int = 200) -> pd.DataFrame:
    t = np.arange(n) * 250_000_000  # 0.25 s stride -> 50 s of windows
    y = np.where(t / 1e9 < 20, 0.16, 0.19)
    return pd.DataFrame({
        "fold_id": "f", "seq_id": "A", "window_id": np.arange(n), "t_start_ns": t,
        "model": "random_forest", "y_true": y, "y_pred": np.full(n, 0.18),
    })


def test_segment_errors_split_opening_from_rest():
    seg = fa.segment_errors(_preds()).set_index("segment")
    assert seg.loc["first_20s", "mean_y_true"] == pytest.approx(0.16)
    assert seg.loc["after_25s", "mean_y_true"] == pytest.approx(0.19)
    assert seg.loc["first_20s", "mae_random_forest"] == pytest.approx(0.02)


def test_spike_stats_constant_offset_is_all_bias():
    p = _preds()
    p["y_pred"] = p["y_true"] + 0.01
    p.loc[0, "y_pred"] += 1e-9  # avoid a zero-variance correlation
    stats = fa.spike_stats(p).iloc[0]
    assert stats["bias2_share_of_mse"] == pytest.approx(1.0, abs=1e-6)
