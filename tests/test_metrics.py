"""
Run with: pytest tests/test_metrics.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import mae, rmse, r2, bias, metrics_table


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.normal(0.18, 0.02, 100)
    p = y + rng.normal(0, 0.01, 100)
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert r2(y, p) == pytest.approx(r2_score(y, p))


def test_perfect_and_constant_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0 and r2(y, y) == 1.0
    assert r2(y, np.full(3, y.mean())) == pytest.approx(0.0)
    assert bias(y, y + 0.5) == pytest.approx(0.5)


def test_metrics_table_has_fold_and_mean_rows():
    preds = pd.DataFrame({
        "model": ["m"] * 4,
        "fold_id": ["f1", "f1", "f2", "f2"],
        "seq_id": ["A", "A", "B", "B"],
        "y_true": [1.0, 2.0, 1.0, 2.0],
        "y_pred": [1.0, 2.0, 2.0, 3.0],
    })
    table = metrics_table(preds).set_index("fold_id")
    assert table.loc["f1", "mae"] == 0.0
    assert table.loc["f2", "mae"] == 1.0
    assert table.loc["mean", "mae"] == 0.5
    assert table.loc["mean", "n"] == 4
