"""
Linear regression trained by mini-batch gradient descent.

OLS, Ridge and Lasso find the weights of ``y_hat = w . x + b`` with a
solver. Here the same model is trained iteratively. Each epoch shuffles the
training windows and splits them into batches of ``batch_size``; for every
batch B the weights take one step against the gradient of the batch loss

    L(w, b) = 1/(2|B|) * sum_{i in B} (y_i - w . x_i - b)^2  +  alpha * R(w)

    w <- w - eta * ( -1/|B| * X_B^T r_B  +  alpha * dR/dw )
    b <- b - eta * ( -1/|B| * sum(r_B) )          (r_B = y_B - X_B w - b)

with R(w) = ||w||^2 / 2 (L2, ``dR/dw = w``) or R(w) = ||w||_1 (L1,
``dR/dw = sign(w)``, a subgradient). The intercept is not penalised.
Batch size sits between the two extremes: 1 is stochastic gradient descent,
the whole training set is batch gradient descent.

Gradient descent is sensitive to feature scale, so features are standardised
first (inside the pipeline, fit on training rows only).

Tuned knobs (see ``PARAM_GRID``): ``learning_rate`` (eta), ``batch_size``,
``penalty`` and ``alpha``. ``loss_curve_`` records the full training loss
after each epoch, so convergence can be plotted.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

PARAM_GRID: dict[str, list] = {
    "gd__learning_rate": [0.001, 0.01],
    "gd__batch_size": [32, 128],
    "gd__penalty": ["l2", "l1"],
    # Penalty on the *mean* batch loss: alpha ~ 1 matches Ridge at alpha ~ 1000
    # on the summed loss with ~1100 training windows.
    "gd__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
}


class MiniBatchGDRegressor(RegressorMixin, BaseEstimator):
    """Squared-loss linear regression fitted by mini-batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 50,
        penalty: str = "l2",
        alpha: float = 1e-3,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha
        self.random_state = random_state

    def _penalty_grad(self, w: np.ndarray) -> np.ndarray:
        if self.penalty == "l2":
            return w
        if self.penalty == "l1":
            return np.sign(w)
        raise ValueError(f"penalty must be 'l2' or 'l1', got {self.penalty!r}")

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        r = y - X @ self.coef_ - self.intercept_
        reg = (0.5 * self.coef_ @ self.coef_ if self.penalty == "l2"
               else np.abs(self.coef_).sum())
        return float(0.5 * np.mean(r * r) + self.alpha * reg)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = float(y.mean())   # start at the mean-baseline prediction
        self.loss_curve_ = [self._loss(X, y)]

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb = X[idx], y[idx]
                r = yb - Xb @ self.coef_ - self.intercept_
                grad_w = -(Xb.T @ r) / len(idx) + self.alpha * self._penalty_grad(self.coef_)
                grad_b = -r.mean()
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b
            self.loss_curve_.append(self._loss(X, y))
        self.n_features_in_ = p
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


def build_minibatch_gd(
    learning_rate: float = 0.01,
    batch_size: int = 32,
    penalty: str = "l2",
    alpha: float = 1e-3,
    random_state: int = 42,
) -> Pipeline:
    """Standardise features, then fit linear regression by mini-batch GD."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gd", MiniBatchGDRegressor(
            learning_rate=learning_rate, batch_size=batch_size,
            penalty=penalty, alpha=alpha, random_state=random_state,
        )),
    ])
