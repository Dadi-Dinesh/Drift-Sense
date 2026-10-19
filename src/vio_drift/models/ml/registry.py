"""
The list of Part D models, in report order, each with its builder and grid.

Scripts and notebooks iterate over ``MODELS`` so every model is trained,
tuned and evaluated the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator

from . import baselines, random_forest


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[..., BaseEstimator]
    param_grid: dict[str, list]


def model_specs(seed: int = 42) -> list[ModelSpec]:
    """All Part D models; ``seed`` fixes the Random Forest's randomness."""
    return [
        ModelSpec("mean", baselines.build_mean_baseline, baselines.MEAN_PARAM_GRID),
        ModelSpec("linear", baselines.build_linear, baselines.LINEAR_PARAM_GRID),
        ModelSpec("ridge", baselines.build_ridge, baselines.RIDGE_PARAM_GRID),
        ModelSpec("lasso", baselines.build_lasso, baselines.LASSO_PARAM_GRID),
        ModelSpec(
            "random_forest",
            lambda: random_forest.build_random_forest(random_state=seed),
            random_forest.PARAM_GRID,
        ),
    ]


MODEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in model_specs())
