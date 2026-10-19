"""
Leave-one-sequence-out must never put the same sequence in train and test.

Run with: pytest tests/test_no_leakage.py -v
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "processed" / "splits.json"


def _splits() -> dict:
    assert SPLITS_PATH.exists(), f"missing {SPLITS_PATH} — Part B should write it"
    return json.loads(SPLITS_PATH.read_text())


def test_splits_file_exists_and_has_three_folds():
    data = _splits()
    assert data["scheme"] == "leave_one_sequence_out"
    assert len(data["folds"]) == 3


def test_no_sequence_in_both_train_and_test():
    data = _splits()
    for fold in data["folds"]:
        train = set(fold["train"])
        test = set(fold["test"])
        overlap = train & test
        assert not overlap, (
            f"{fold['fold_id']}: sequence(s) {sorted(overlap)} appear in "
            f"both train and test — that leaks overlapping windows"
        )


def test_each_sequence_is_held_out_exactly_once():
    data = _splits()
    sequences = set(data["sequences"])
    held_out = []
    for fold in data["folds"]:
        assert len(fold["test"]) == 1
        held_out.extend(fold["test"])
        assert set(fold["train"]) | set(fold["test"]) == sequences
    assert sorted(held_out) == sorted(sequences)


def test_primary_target_is_raw_magnitude():
    data = _splits()
    assert data["primary_target"] == "err_mag_m"


def test_chronological_split_leaves_no_overlapping_windows():
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from vio_drift.models.ml.data import chronological_split

    window_ns, stride_ns = int(1.0e9), int(0.25e9)
    flight = pd.DataFrame({"t_start_ns": np.arange(400) * stride_ns})
    train, test = chronological_split(flight, train_frac=0.6, gap_windows=8)
    assert len(train) + len(test) + 8 == len(flight)
    # the last training window must end before the first test window starts
    assert train["t_start_ns"].max() + window_ns <= test["t_start_ns"].min()
