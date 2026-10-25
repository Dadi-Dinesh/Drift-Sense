"""
Read the leave-one-sequence-out folds in ``data/processed/splits.json``.

Folds are re-checked on load: a fold whose train and test sequences overlap
raises instead of silently leaking overlapping windows into the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fold:
    """One outer cross-validation fold: train on some sequences, test on one."""
    fold_id: str
    train: tuple[str, ...]
    test: tuple[str, ...]


def load_splits(splits_path: Path) -> list[Fold]:
    """Read leave-one-sequence-out folds from ``splits.json`` and validate them."""
    data = json.loads(Path(splits_path).read_text())
    folds = [
        Fold(fold_id=f["fold_id"], train=tuple(f["train"]), test=tuple(f["test"]))
        for f in data["folds"]
    ]
    for fold in folds:
        overlap = set(fold.train) & set(fold.test)
        if overlap:
            raise ValueError(f"{fold.fold_id}: {sorted(overlap)} in both train and test")
    return folds
