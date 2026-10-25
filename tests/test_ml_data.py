"""
Checks for Part D's data loading on the real feature tables.

Run with: pytest tests/test_ml_data.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.data.splits import load_splits
from vio_drift.features.build_features import feature_names
from vio_drift.models.ml.data import fold_frames, load_dataset

PROCESSED = ROOT / "data" / "processed"
SEQUENCES = ["MH_01_easy", "MH_02_easy"]

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "features_MH_01_easy.csv").exists(),
    reason="Part C feature tables not built",
)


def test_dataset_has_one_row_per_window_with_features_and_target():
    df = load_dataset(PROCESSED, SEQUENCES, target="err_mag_m")
    assert len(df) == 720 + 592
    assert list(df.columns) == ["seq_id", "window_id", "t_start_ns",
                                *feature_names(), "err_mag_m"]
    assert not df.isna().any().any()


def test_rows_are_in_time_order_within_each_flight():
    df = load_dataset(PROCESSED, SEQUENCES, target="err_mag_m")
    assert df.groupby("seq_id")["t_start_ns"].apply(lambda s: s.is_monotonic_increasing).all()


def test_any_label_column_can_be_the_target():
    df = load_dataset(PROCESSED, SEQUENCES, target="err_dx")
    assert df.columns[-1] == "err_dx"


def test_folds_keep_the_test_flight_out_of_train():
    df = load_dataset(PROCESSED, SEQUENCES, target="err_mag_m")
    for fold in load_splits(PROCESSED / "splits.json"):
        train, test = fold_frames(df, fold)
        assert set(train["seq_id"]) == set(fold.train)
        assert set(test["seq_id"]) == set(fold.test)
        assert len(train) + len(test) == len(df)
