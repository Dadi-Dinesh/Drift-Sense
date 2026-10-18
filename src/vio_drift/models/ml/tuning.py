"""
Hyperparameter search that never lets the test sequence leak in.

Evaluation is *nested*:

  * The **outer** loop is the leave-one-sequence-out folds in
    ``splits.json``. Each fold holds one sequence out for the final score.
  * The **inner** loop (this module) tunes hyperparameters using only that
    fold's training sequences. It again holds out one whole sequence at a
    time (``LeaveOneGroupOut`` with ``groups = seq_id``), because windows
    overlap by 75% — a random inner split would score candidates on windows
    nearly identical to ones they trained on and pick over-fitted settings.

With two training sequences, the inner loop is "train on A, score on B"
and "train on B, score on A". The winning setting is then refit on both and
scored once on the held-out outer sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

SCORING = "neg_mean_absolute_error"


@dataclass
class TuningResult:
    """Outcome of tuning one model on one outer fold's training sequences."""
    best_estimator: BaseEstimator   # refit on all training rows
    best_params: dict
    inner_cv_mae: float             # mean MAE over the inner held-out sequences
    cv_results: pd.DataFrame        # one row per candidate setting


def tune(
    estimator: BaseEstimator,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_jobs: int = -1,
) -> TuningResult:
    """
    Grid-search ``param_grid`` with sequence-grouped CV, then refit the best.

    ``groups`` must hold each row's ``seq_id``; at least two distinct
    sequences are required so every inner score comes from an unseen one.
    An empty ``param_grid`` still runs the inner CV (to report a comparable
    ``inner_cv_mae``) and refits the estimator as-is.
    """
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(
            f"sequence-grouped CV needs >= 2 training sequences, got {n_groups}"
        )

    search = GridSearchCV(
        clone(estimator),
        param_grid=param_grid or [{}],
        scoring=SCORING,
        cv=LeaveOneGroupOut(),
        refit=True,
        n_jobs=n_jobs,
    )
    search.fit(X, y, groups=groups)

    cv_results = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    cv_results = cv_results[keep].assign(mean_test_mae=-cv_results["mean_test_score"])

    return TuningResult(
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        inner_cv_mae=float(-search.best_score_),
        cv_results=cv_results,
    )
