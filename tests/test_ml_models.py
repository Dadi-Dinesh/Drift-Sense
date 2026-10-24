"""
Toy-data checks for Part D models and metrics. Does not need EuRoC.

Run with: pytest tests/test_ml_models.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.metrics import score
from vio_drift.models.ml.models import MODELS


def _toy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)) * [1, 10, 100, 1000], columns=list("abcd"))
    y = 1.2 + 0.05 * X["a"].to_numpy() + rng.normal(0, 0.005, n)
    return X, y


def test_score_matches_sklearn():
    y = np.array([1.0, 1.2, 1.1, 1.4])
    p = np.array([1.1, 1.1, 1.2, 1.3])
    s = score(y, p)
    assert np.isclose(s["mae"], mean_absolute_error(y, p))
    assert np.isclose(s["rmse"], np.sqrt(mean_squared_error(y, p)))
    assert np.isclose(s["r2"], r2_score(y, p))
    assert np.isclose(s["bias"], np.mean(p - y))


def test_mean_baseline_predicts_training_mean():
    X, y = _toy()
    model = MODELS["mean"]().fit(X, y)
    assert np.allclose(model.predict(X), y.mean())


def test_ols_learns_a_linear_target_and_beats_the_mean():
    X, y = _toy()
    ols = MODELS["ols"]().fit(X[:150], y[:150])
    mean = MODELS["mean"]().fit(X[:150], y[:150])
    assert score(y[150:], ols.predict(X[150:]))["r2"] > 0.9
    assert score(y[150:], mean.predict(X[150:]))["r2"] < 0.05


def test_ols_scaler_is_fit_on_training_rows_only():
    X, y = _toy()
    ols = MODELS["ols"]().fit(X[:150], y[:150])
    scaler = ols.named_steps["scale"]
    assert np.allclose(scaler.mean_, X[:150].mean().to_numpy())
