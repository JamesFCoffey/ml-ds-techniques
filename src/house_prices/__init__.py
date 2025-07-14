"""
house_prices package.

This package provides a unified interface for:
  - Data featurization via :func:`featurize.make_preprocessor`
  - Objective functions for hyperparameter tuning of various models:
      * CART (:func:`train.make_cart_objective`)
      * Random Forest (:func:`train.make_rf_objective`)
      * Gradient Boosted Trees (:func:`train.make_gbts_objective`)
      * XGBoost (:func:`train.make_xgb_objective`)
      * Multi-Layer Perceptron (:func:`train.make_mlp_objective`)

The module exports the key factory functions to build Optuna objectives
and the preprocessing pipeline, so that notebooks and scripts can
simply import from this package without referencing submodules directly.

Example:
    >>> from house_prices import make_preprocessor, make_gbts_objective
    >>> X_train, X_valid, y_train, y_valid, df_tr, df_val, preproc = \
    ...     make_preprocessor(df_raw)
    >>> gbts_obj = make_gbts_objective(df_tr, df_val, y_valid)
    >>> study = optuna.create_study(direction="minimize")
    >>> study.optimize(gbts_obj, n_trials=20)

Attributes:
    make_preprocessor (callable): Build and return the preprocessing pipeline
        and split data for modeling.
    make_cart_objective (callable): Factory for CART Optuna objective.
    make_rf_objective (callable): Factory for Random Forest Optuna objective.
    make_gbts_objective (callable): Factory for GBT Optuna objective.
    make_xgb_objective (callable): Factory for XGBoost Optuna objective.
    make_mlp_objective (callable): Factory for MLP Optuna objective.
"""

from .featurize import make_preprocessor
from .train import (
    make_cart_objective,
    make_gbts_objective,
    make_mlp_objective,
    make_rf_objective,
    make_xgb_objective,
)

__all__ = [
    "make_preprocessor",
    "make_cart_objective",
    "make_gbts_objective",
    "make_mlp_objective",
    "make_rf_objective",
    "make_xgb_objective",
]
