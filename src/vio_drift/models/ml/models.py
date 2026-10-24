"""
Part D models. Each entry in ``MODELS`` builds a fresh, unfitted estimator.

  mean  predicts the training flight's mean drift for every window; any
        useful model must beat it
  ols   ordinary least squares on standardised features

``build_ridge`` (L2) and ``build_lasso`` (L1) take the penalty strength
``alpha``; they join ``MODELS`` once a way to choose ``alpha`` is settled.

Features are standardised inside the pipeline, so the scaler's mean and
std come from the training flight only and the test flight never leaks in.
"""
from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mean() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def build_ols() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])


def build_ridge(alpha: float) -> Pipeline:
    """L2 penalty: shrinks every weight towards zero, keeps all features."""
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def build_lasso(alpha: float) -> Pipeline:
    """L1 penalty: pushes weak features' weights to exactly zero."""
    return Pipeline([("scale", StandardScaler()),
                     ("model", Lasso(alpha=alpha, max_iter=50_000))])


MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "mean": build_mean,
    "ols": build_ols,
}
