"""
Fit one Part D model on a training flight, tuning alpha first if it has one.

Shared by scripts/train_eval.py and notebooks 05/06 so they train exactly
the same way: models with an ``alphas`` grid pick alpha by time-block CV on
the training rows (median rule, see vio_drift.models.ml.tuning) and are
then refit on all of them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vio_drift.models.ml.models import MODELS
from vio_drift.models.ml.tuning import GAP_WINDOWS, tune_alpha


@dataclass
class FittedModel:
    model: BaseEstimator
    alpha: float | None                 # None for models with nothing to tune
    cv_table: pd.DataFrame | None       # every alpha's CV MAE per block


def fit_model(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    n_blocks: int = 5,
    gap_windows: int = GAP_WINDOWS,
) -> FittedModel:
    """Tune (if needed) and fit ``MODELS[name]`` on time-ordered training rows."""
    spec = MODELS[name]
    alpha, cv_table = None, None
    if spec.alphas is None:
        model = spec.build()
    else:
        tuned = tune_alpha(spec.build, spec.alphas, X, y, n_blocks, gap_windows)
        alpha, cv_table = tuned.median_alpha, tuned.cv_table
        model = spec.build(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # Lasso convergence at tiny alpha
        model.fit(X, y)
    return FittedModel(model, alpha, cv_table)
