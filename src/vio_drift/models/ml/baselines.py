"""
Reference models every real model must beat.

* **Mean baseline** — ignores the features and always predicts the average
  label of the training sequences. If a model cannot beat this, its features
  carry no usable information about drift on an unseen sequence.
* **Linear regression** — the same straight-line fit ``y = w . x + b`` with
  no penalty: ordinary least squares. Kept as the unregularised reference so
  the effect of Ridge's penalty can be measured rather than assumed.
* **Ridge regression** — a straight-line fit ``y = w . x + b`` with an L2
  penalty ``alpha * ||w||^2`` that shrinks weights. The penalty matters here:
  many of the 104 features are near-duplicates (e.g. ``var = std^2``), and an
  unpenalised linear fit would assign them huge, cancelling weights. Features
  are standardised inside the pipeline so the penalty treats them equally and
  the scaler is fit on training rows only.
* **Lasso regression** — the same fit with an L1 penalty ``alpha * sum |w|``.
  Unlike L2, L1 drives weak weights to exactly zero, so Lasso also selects
  features: the surviving non-zero weights show which features it relies on.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MEAN_PARAM_GRID: dict[str, list] = {}

LINEAR_PARAM_GRID: dict[str, list] = {}  # nothing to tune

RIDGE_PARAM_GRID: dict[str, list] = {
    "ridge__alpha": [float(a) for a in np.logspace(-2, 5, 8)],  # 0.01 ... 1e5
}

# The label's spread is ~0.015 m, so useful L1 strengths are small: around
# 1e-2 and above every weight is already zero (the model predicts the mean).
LASSO_PARAM_GRID: dict[str, list] = {
    "lasso__alpha": [float(a) for a in np.logspace(-6, -1, 6)],  # 1e-6 ... 0.1
}


def build_mean_baseline() -> DummyRegressor:
    """Predict the training-set mean for every window."""
    return DummyRegressor(strategy="mean")


def build_linear() -> Pipeline:
    """Standardise features, then fit ordinary least squares (no penalty)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression()),
    ])


def build_ridge(alpha: float = 1.0) -> Pipeline:
    """Standardise features, then fit L2-penalised linear regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_lasso(alpha: float = 1e-4) -> Pipeline:
    """Standardise features, then fit L1-penalised linear regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        # Collinear features slow coordinate descent; allow more iterations.
        ("lasso", Lasso(alpha=alpha, max_iter=20_000)),
    ])
