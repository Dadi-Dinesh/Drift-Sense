"""
Checks for Part D's time-block CV and alpha tuning. Does not need EuRoC.

Run with: pytest tests/test_ml_tuning.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.models.ml.models import build_lasso, build_ridge
from vio_drift.models.ml.tuning import time_block_splits, tune_alpha


def test_blocks_are_contiguous_and_cover_every_row_once():
    splits = time_block_splits(300, n_blocks=5, gap_windows=16)
    assert len(splits) == 5
    scored = np.concatenate([test for _, test in splits])
    assert sorted(scored) == list(range(300))
    for _, test in splits:
        assert np.all(np.diff(test) == 1)


def test_gap_keeps_training_rows_away_from_the_scored_block():
    for train, test in time_block_splits(300, n_blocks=5, gap_windows=16):
        dist = np.minimum(np.abs(train[:, None] - test[None, :]), 10**9).min()
        assert dist > 16


def test_too_many_blocks_for_the_data_raises():
    with pytest.raises(ValueError):
        time_block_splits(50, n_blocks=10, gap_windows=16)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"f{i}" for i in range(20)])
    y = 1.2 + 0.05 * X["f0"].to_numpy() + rng.normal(0, 0.01, n)
    return X, y


def test_tune_alpha_picks_from_the_grid_and_reports_every_candidate():
    X, y = _toy()
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    result = tune_alpha(build_lasso, alphas, X, y, n_blocks=5, gap_windows=4)
    assert result.best_alpha in alphas
    assert result.one_se_alpha in alphas
    assert result.one_se_alpha >= result.best_alpha
    assert list(result.cv_table["alpha"]) == alphas


def test_tune_alpha_avoids_alpha_that_kills_the_signal():
    X, y = _toy()
    result = tune_alpha(build_lasso, [1e-3, 1.0], X, y, n_blocks=5, gap_windows=4)
    assert result.best_alpha == 1e-3          # alpha=1 zeroes every weight


def test_tune_alpha_works_for_ridge():
    X, y = _toy()
    result = tune_alpha(build_ridge, [0.1, 10.0, 1e6], X, y, n_blocks=5, gap_windows=4)
    assert result.best_alpha < 1e6
