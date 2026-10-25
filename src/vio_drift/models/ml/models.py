"""
Part D models, in report order. Each ``ModelSpec`` builds a fresh, unfitted
estimator; models with an ``alphas`` grid take ``alpha`` and are tuned on the
training flight (vio_drift.models.ml.tuning) before the final fit.

  mean   predicts the training flight's mean drift for every window; any
         useful model must beat it
  ols    ordinary least squares on standardised features
  ridge  OLS + L2 penalty
  lasso  OLS + L1 penalty

The grids span "no penalty" to "every weight ~0" on standardised features
(see results/diagnostics/alpha_sweep_err_mag_m.csv and figures/ml/alpha_sweep_*.png).

Features are standardised inside the pipeline, so the scaler's mean and
std come from the training flight only and the test flight never leaks in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
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


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[..., BaseEstimator]
    alphas: list[float] | None = None     # None: nothing to tune


MODELS: dict[str, ModelSpec] = {
    "mean": ModelSpec(build_mean),
    "ols": ModelSpec(build_ols),
    "ridge": ModelSpec(build_ridge, [float(a) for a in np.logspace(-3, 5, 17)]),
    "lasso": ModelSpec(build_lasso, [float(a) for a in np.logspace(-6, -1, 16)]),
}
