"""
Checks the saved final model (scripts/train_final.py) loads and predicts.

Run with: pytest tests/test_final_model.py -v
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.features.build_features import feature_names

PKL = ROOT / "models" / "ml" / "lasso_err_mag_m.pkl"
META = PKL.with_suffix(".json")

pytestmark = pytest.mark.skipif(not PKL.exists(), reason="run scripts/train_final.py first")


def _load():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def test_model_expects_the_104_features_in_order():
    model = _load()
    assert list(model.feature_names_in_) == feature_names()
    assert json.loads(META.read_text())["feature_order"] == feature_names()


def test_metadata_matches_the_pickled_model():
    model, meta = _load(), json.loads(META.read_text())
    assert model.named_steps["model"].alpha == meta["alpha"]
    kept = [f for f, c in zip(feature_names(), model.named_steps["model"].coef_) if c != 0]
    assert kept == meta["features_kept"]


def test_predictions_on_a_feature_table_are_plausible_drift():
    model = _load()
    features = pd.read_csv(ROOT / "data" / "processed" / "features_MH_01_easy.csv")
    drift = model.predict(features[model.feature_names_in_])
    assert len(drift) == len(features)
    assert np.all(np.isfinite(drift))
    assert 0.8 < drift.mean() < 1.6                 # metres, ~1.2 m per 2 s window
