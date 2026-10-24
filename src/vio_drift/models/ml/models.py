"""
Part D models. Each entry in ``MODELS`` builds a fresh, unfitted estimator.

  mean  predicts the training flight's mean drift for every window; any
        useful model must beat it
  ols   ordinary least squares on standardised features

Features are standardised inside the pipeline, so the scaler's mean and
std come from the training flight only and the test flight never leaks in.
"""
from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mean() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def build_ols() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])


MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "mean": build_mean,
    "ols": build_ols,
}
