"""
Toy-data checks for Part D models and sequence-grouped tuning. Does not
need EuRoC; the last test uses the real feature tables if present.

Run with: pytest tests/test_ml_models.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.models.ml.data import load_splits, load_dataset, fold_frames
from vio_drift.models.ml.registry import model_specs
from vio_drift.models.ml.tuning import inner_splits, tune

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEQUENCES = ["MH_01_easy", "MH_02_easy"]


def _toy(n_per_seq: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 3 * n_per_seq
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = 0.2 + 0.05 * X["a"].to_numpy() + rng.normal(0, 0.005, n)
    groups = np.repeat(["S1", "S2", "S3"], n_per_seq)
    return X, y, groups


def test_mean_baseline_predicts_training_mean():
    X, y, _ = _toy()
    spec = next(s for s in model_specs() if s.name == "mean")
    model = spec.build().fit(X, y)
    assert np.allclose(model.predict(X), y.mean())


@pytest.mark.parametrize("name", ["linear", "ridge", "lasso", "random_forest"])
def test_models_beat_mean_on_learnable_toy_data(name):
    X, y, _ = _toy()
    spec = next(s for s in model_specs() if s.name == name)
    model = spec.build().fit(X[:120], y[:120])
    err = np.mean(np.abs(model.predict(X[120:]) - y[120:]))
    err_mean = np.mean(np.abs(y[:120].mean() - y[120:]))
    assert err < err_mean


def test_tune_uses_whole_sequences_as_inner_folds():
    X, y, groups = _toy()
    spec = next(s for s in model_specs() if s.name == "ridge")
    result = tune(spec.build(), spec.param_grid, X, y, groups, n_jobs=1)
    assert set(result.best_params) == set(spec.param_grid)
    assert len(result.cv_results) == len(spec.param_grid["ridge__alpha"])
    assert result.inner_cv_mae > 0


def test_tune_handles_empty_grid():
    X, y, groups = _toy()
    spec = next(s for s in model_specs() if s.name == "mean")
    result = tune(spec.build(), spec.param_grid, X, y, groups, n_jobs=1)
    assert result.best_params == {}


def test_inner_splits_hold_out_whole_sequences():
    _, _, groups = _toy()
    splits = inner_splits(groups)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(set(groups[test_idx])) == 1
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_inner_splits_purge_a_gap_around_each_time_block():
    groups = np.full(300, "S1")
    splits = inner_splits(groups, n_blocks=3, gap_windows=16)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert np.all(np.diff(test_idx) == 1)          # one contiguous block
        lo, hi = test_idx[0], test_idx[-1]
        dist = np.minimum(np.abs(train_idx - lo), np.abs(train_idx - hi))
        assert dist.min() > 16
    assert sorted(np.concatenate([t for _, t in splits])) == list(range(300))


def test_tune_works_on_a_single_sequence():
    X, y, _ = _toy()
    spec = next(s for s in model_specs() if s.name == "ridge")
    result = tune(spec.build(), spec.param_grid, X, y, np.full(len(y), "S1"),
                  n_jobs=1, gap_windows=4)
    assert set(result.best_params) == set(spec.param_grid)
    assert result.inner_cv_mae > 0


@pytest.mark.skipif(not (PROCESSED / "features_MH_01_easy.csv").exists(),
                    reason="Part C feature tables not built")
def test_real_folds_keep_test_sequence_out_of_train():
    df = load_dataset(PROCESSED, SEQUENCES, target="err_mag_m")
    assert df.isna().sum().sum() == 0
    for fold in load_splits(PROCESSED / "splits.json"):
        train, test = fold_frames(df, fold)
        assert set(train["seq_id"]).isdisjoint(test["seq_id"])
        assert set(test["seq_id"]) == set(fold.test)
        assert len(train) + len(test) == len(df)


def test_lasso_zeroes_irrelevant_features():
    X, y, _ = _toy()                       # only column "a" drives y
    spec = next(s for s in model_specs() if s.name == "lasso")
    model = spec.build().set_params(lasso__alpha=1e-3).fit(X, y)
    w = model[-1].coef_
    assert w[0] != 0.0 and np.all(w[1:] == 0.0)

