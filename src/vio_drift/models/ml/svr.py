"""
Support Vector Regression (RBF kernel) for window drift.

SVR fits a function ``f(x) = sum_i a_i K(x_i, x) + b`` that keeps as many
training labels as possible inside a tube of half-width ``epsilon`` around
it. Errors smaller than ``epsilon`` cost nothing; larger errors cost
linearly, weighted by ``C``. Only the training points on or outside the
tube (the *support vectors*) end up with non-zero ``a_i``.

The RBF kernel ``K(x, x') = exp(-gamma * ||x - x'||^2)`` measures similarity
between two windows; ``gamma`` sets how quickly similarity decays with
distance (large gamma = very local, wiggly fit).

Two scaling steps are essential, both fit on training rows only:
  * **Features** are standardised, because the kernel uses raw Euclidean
    distance — an unscaled feature in m/s^3 (jerk) would swamp one in rad/s.
  * **The target** is standardised too. ``err_mag_m`` is about 0.185 m with
    a spread of ~0.015 m, so sklearn's default ``epsilon = 0.1`` (in label
    units) would put every window inside the tube and learn nothing. With a
    standardised target, ``epsilon`` and ``C`` are in units of label
    standard deviations and the same grid works for any target.
"""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Keys reach through TransformedTargetRegressor (regressor__) into the
# pipeline step named "svr".
PARAM_GRID: dict[str, list] = {
    "regressor__svr__C": [0.1, 1.0, 10.0],
    "regressor__svr__gamma": ["scale", 0.001, 0.01],
    "regressor__svr__epsilon": [0.1, 0.5],
}


def build_svr(
    C: float = 1.0,
    gamma: float | str = "scale",
    epsilon: float = 0.1,
) -> TransformedTargetRegressor:
    """Standardise X and y, then fit an RBF-kernel SVR."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)),
    ])
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
