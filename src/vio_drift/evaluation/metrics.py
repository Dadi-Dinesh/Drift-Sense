"""
Regression metrics for Part D, all in metres except R^2.

  mae   mean absolute error — the headline number
  rmse  root mean squared error — punishes large misses more
  r2    1 - SSE/SST on the test flight; 0 = no better than predicting the
        test flight's own mean, < 0 = worse than that
  bias  mean(prediction - truth); > 0 means the model over-predicts drift
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R^2 and bias of ``y_pred`` against ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }
