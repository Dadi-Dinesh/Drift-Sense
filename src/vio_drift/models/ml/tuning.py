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

With one training sequence (Phase 1 has only MH_01 and MH_02, so every
outer fold trains on a single flight) there is no other sequence to hold
out. The flight is then cut into contiguous time blocks; each block is
scored in turn by a model trained on the others, with ``gap_windows``
windows dropped on either side of the held-out block so no training window
shares samples with a scored one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from vio_drift.models.ml.data import GAP_WINDOWS

SCORING = "neg_mean_absolute_error"


@dataclass
class TuningResult:
    """Outcome of tuning one model on one outer fold's training sequences."""
    best_estimator: BaseEstimator   # refit on all training rows
    best_params: dict
    inner_cv_mae: float             # mean MAE over the inner held-out folds
    cv_results: pd.DataFrame        # one row per candidate setting


def inner_splits(
    groups: np.ndarray,
    n_blocks: int = 3,
    gap_windows: int = GAP_WINDOWS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Inner CV folds as ``(train_idx, test_idx)`` pairs.

    Two or more sequences: hold out one whole sequence at a time. One
    sequence: hold out each of ``n_blocks`` contiguous time blocks, purging
    ``gap_windows`` rows either side of it from training. Rows must be in
    time order within the sequence, as ``load_dataset`` returns them.
    """
    groups = np.asarray(groups)
    if len(np.unique(groups)) >= 2:
        return list(LeaveOneGroupOut().split(groups, groups=groups))

    idx = np.arange(len(groups))
    splits = []
    for block in np.array_split(idx, n_blocks):
        lo, hi = block[0], block[-1]
        keep = (idx < lo - gap_windows) | (idx > hi + gap_windows)
        splits.append((idx[keep], block))
    return splits


def tune(
    estimator: BaseEstimator,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_jobs: int = -1,
    gap_windows: int = GAP_WINDOWS,
) -> TuningResult:
    """
    Grid-search ``param_grid`` with sequence-grouped CV, then refit the best.

    ``groups`` must hold each row's ``seq_id``. With a single sequence the
    inner folds are purged time blocks (see ``inner_splits``). An empty ``param_grid`` still runs the inner CV (to report a comparable
    ``inner_cv_mae``) and refits the estimator as-is.
    """
    search = GridSearchCV(
        clone(estimator),
        param_grid=param_grid or [{}],
        scoring=SCORING,
        cv=inner_splits(groups, gap_windows=gap_windows),
        refit=True,
        n_jobs=n_jobs,
    )
    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    cv_results = cv_results[keep].assign(mean_test_mae=-cv_results["mean_test_score"])

    return TuningResult(
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        inner_cv_mae=float(-search.best_score_),
        cv_results=cv_results,
    )
