"""
Random Forest regressor for window drift.

A regression tree splits the feature space with yes/no questions
("is ax_rms > 9.6?") chosen to minimise the squared error of the label in
each resulting group, and predicts the mean label of the leaf a window falls
into. One deep tree memorises noise, so a forest trains many trees, each on
a bootstrap resample of the rows and considering only a random subset of
features at each split (``max_features``), and averages their predictions.
Averaging many de-correlated trees cuts variance without adding much bias.

No scaling is needed: tree splits depend only on the order of a feature's
values, not their units.

Tuned knobs (see ``PARAM_GRID``):
  * ``max_depth`` / ``min_samples_leaf`` — how finely a tree may split;
    shallower trees with bigger leaves generalise better on small data.
  * ``max_features`` — features tried per split; fewer = more de-correlated
    trees.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

DEFAULT_N_ESTIMATORS = 300

PARAM_GRID: dict[str, list] = {
    "max_depth": [None, 4, 8],
    "min_samples_leaf": [1, 5, 20],
    "max_features": [1.0, "sqrt", 0.3],
}


def build_random_forest(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
    max_features: float | str = 0.3,
    random_state: int = 42,
    n_jobs: int = 1,
) -> RandomForestRegressor:
    """
    Random Forest with project defaults.

    ``n_jobs`` defaults to 1 because grid search already parallelises over
    candidate settings; nesting both would oversubscribe the CPU.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
