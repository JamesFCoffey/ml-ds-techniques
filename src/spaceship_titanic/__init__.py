"""Public API for the *spaceship_titanic* package.

This package bundles end-to-end utilities for the Kaggle *Spaceship Titanic*
competition:

* **Feature engineering:** :class:`~spaceship_titanic.SpaceshipTransformer`.
* **Training helpers:** :func:`~spaceship_titanic.make_pipe`,
  :func:`~spaceship_titanic.make_fold_data`.
* **Hyper-parameter tuning objectives** for CatBoost/LightGBM/XGBoost.
* **Ensembling helpers** for cross-validated folds and weight search.

Importing the package exposes the most common entry points::

    >>> from spaceship_titanic import SpaceshipTransformer, make_pipe
"""

from .ensemble import run_fold, weight_objective
from .featurize import (
    GROUP_RE,
    SpaceshipTransformer,
)
from .train import make_fold_data, make_pipe
from .tune import objective_cat, objective_lgb, objective_xgb

__all__ = [
    "GROUP_RE",
    "SpaceshipTransformer",
    "make_pipe",
    "make_fold_data",
    "objective_cat",
    "objective_lgb",
    "objective_xgb",
    "run_fold",
    "weight_objective",
]
