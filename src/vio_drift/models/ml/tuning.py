"""
Choose a model's penalty ``alpha`` using the training flight only.

Each fold has a single training flight, so there is no second flight to
validate on. Instead the flight is cut into ``n_blocks`` contiguous time
blocks; each block is scored by a model trained on the others. Windows
overlap by 87.5%, so a random split would score windows nearly identical to
training ones; blocks keep scored and training windows apart in time, and
``gap_windows`` rows either side of the scored block are dropped from
training so none of them share IMU samples.

Three picking rules are reported:
  median_alpha  lowest *median* block MAE — the rule Part D uses
  best_alpha    lowest mean CV MAE
  one_se_alpha  the largest (most regularised) alpha whose mean CV MAE is
                within one standard error of the best — the 1-SE rule

The median rule is used because the first block of each flight is the
take-off: drift there is lower and 3-5x more variable than in the rest of
the flight. When that block is scored, the model has seen no take-off at
all and misses by 10+ cm, so it dominates the mean and drives the mean and
1-SE rules towards "predict the average" (huge alpha). The final model is
trained on the whole flight, take-off included, so that block's score is
unrepresentative; the median lets the other blocks decide.

``tune_alpha_by_flight`` is for the final model trained on *both* flights:
there each flight can be held out in turn, so alpha is chosen by
leave-one-flight-out CV (mean MAE over the two held-out flights).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

# 16 windows = 4 s at 0.25 s stride: twice the 2 s window length, so two
# windows this far apart share no IMU samples.
GAP_WINDOWS = 16


@dataclass
class TuningResult:
    median_alpha: float
    best_alpha: float
    one_se_alpha: float
    cv_table: pd.DataFrame      # alpha, mean_mae, se_mae, one column per block


def time_block_splits(
    n_rows: int,
    n_blocks: int = 5,
    gap_windows: int = GAP_WINDOWS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    ``(train_idx, score_idx)`` pairs: each of ``n_blocks`` contiguous blocks
    is scored once, with ``gap_windows`` rows either side purged from training.
    Rows must be in time order.
    """
    if n_rows // n_blocks <= 2 * gap_windows:
        raise ValueError(f"{n_rows} rows are too few for {n_blocks} blocks "
                         f"with a {gap_windows}-row gap")
    idx = np.arange(n_rows)
    splits = []
    for block in np.array_split(idx, n_blocks):
        keep = (idx < block[0] - gap_windows) | (idx > block[-1] + gap_windows)
        splits.append((idx[keep], block))
    return splits


def median_rule(cv_table: pd.DataFrame) -> float:
    """The alpha with the lowest median MAE across the ``block_*`` columns."""
    medians = cv_table.filter(like="block_").median(axis=1)
    return float(cv_table.loc[medians.idxmin(), "alpha"])


def tune_alpha(
    build: Callable[[float], BaseEstimator],
    alphas: list[float],
    X: pd.DataFrame,
    y: np.ndarray,
    n_blocks: int = 5,
    gap_windows: int = GAP_WINDOWS,
) -> TuningResult:
    """Time-block CV MAE for every ``alpha``; returns all three picking rules."""
    y = np.asarray(y)
    splits = time_block_splits(len(X), n_blocks, gap_windows)
    rows = []
    for alpha in alphas:
        maes = []
        for train_idx, score_idx in splits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # Lasso convergence at tiny alpha
                model = build(alpha).fit(X.iloc[train_idx], y[train_idx])
            maes.append(mean_absolute_error(y[score_idx], model.predict(X.iloc[score_idx])))
        rows.append({"alpha": alpha,
                     "mean_mae": float(np.mean(maes)),
                     "se_mae": float(np.std(maes, ddof=1) / np.sqrt(len(maes))),
                     **{f"block_{i}": m for i, m in enumerate(maes)}})
    table = pd.DataFrame(rows)

    best = table.loc[table["mean_mae"].idxmin()]
    within = table[table["mean_mae"] <= best["mean_mae"] + best["se_mae"]]
    return TuningResult(median_alpha=median_rule(table),
                        best_alpha=float(best["alpha"]),
                        one_se_alpha=float(within["alpha"].max()),
                        cv_table=table)


@dataclass
class FlightTuningResult:
    best_alpha: float
    cv_table: pd.DataFrame      # alpha, mean_mae, one column per held-out flight


def tune_alpha_by_flight(
    build: Callable[[float], BaseEstimator],
    alphas: list[float],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> FlightTuningResult:
    """Leave-one-flight-out CV MAE for every ``alpha``; picks the lowest mean."""
    y, groups = np.asarray(y), np.asarray(groups)
    flights = list(dict.fromkeys(groups))
    if len(flights) < 2:
        raise ValueError(f"need at least two flights, got {flights}")
    rows = []
    for alpha in alphas:
        maes = {}
        for flight in flights:
            held = groups == flight
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # Lasso convergence at tiny alpha
                model = build(alpha).fit(X[~held], y[~held])
            maes[f"flight_{flight}"] = mean_absolute_error(y[held], model.predict(X[held]))
        rows.append({"alpha": alpha, "mean_mae": float(np.mean(list(maes.values()))), **maes})
    table = pd.DataFrame(rows)
    return FlightTuningResult(best_alpha=float(table.loc[table["mean_mae"].idxmin(), "alpha"]),
                              cv_table=table)
