"""Hyper-parameter optimization helpers for the Titanic project.

This module centralizes **all Optuna-related glue** so notebooks and CLI scripts
can import a single, well-tested API.

Main capabilities
-----------------
* ``make_convergence_callback`` – build a generic early-stopping / pruning
  callback that halts a study when progress stalls.
* ``make_gbts_cv_objective``  – factory that returns an Optuna objective for a
  YDF Gradient-Boosted-Trees learner, evaluated with leak-free stratified K-fold
  cross-validation.
* ``make_xgb_cv_objective``   – same idea for an XGBoost classifier, with
  automatic GPU detection.

All objectives return **mean fold accuracy** so they can be maximized directly
by Optuna.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import ydf
from optuna.study import StudyDirection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .featurize import preprocess_fold
from .models import YDFWrapper  # noqa: F401  (imported for type‐checking)


# ----------------------------------------------------------------------
#  Generic convergence / pruning callback
# ----------------------------------------------------------------------
def make_convergence_callback(
    start_trial: int = 200,
    lookback: int = 100,
    epsilon: float = 1e-4,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Return an Optuna callback that stops a study once it converges.

    Convergence is declared when the best value has **not** improved by at least
    ``epsilon`` over the last ``lookback`` *completed* trials. The first
    ``start_trial`` completed trials are always ignored so the optimizer has
    time to explore.

    Args:
        start_trial: Number of completed trials to wait before monitoring
            convergence.
        lookback: Size of the moving window (in *completed* trials) that
            is inspected for progress.
        epsilon: Minimum absolute improvement that must be observed
            within the look-back window; if not met, the study is stopped via
            :py:meth:`optuna.study.Study.stop`.

    Returns:
        A callback compatible with Optuna’s ``callbacks`` API.  Pass it directly
        to :py:meth:`optuna.study.Study.optimize`.
    """

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:  # noqa: D401,E501
        """Internal closure executed by Optuna after every trial."""
        finished = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(finished) < max(start_trial, lookback + 1):
            return  # not enough data yet

        # Sort to ensure deterministic window selection
        finished.sort(key=lambda t: t.number)

        if study.direction == StudyDirection.MINIMIZE:
            best_before = min(t.value for t in finished[:-lookback])
            best_now = min(t.value for t in finished)
            improvement = best_before - best_now  # positive => better
        else:  # MAXIMIZE
            best_before = max(t.value for t in finished[:-lookback])
            best_now = max(t.value for t in finished)
            improvement = best_now - best_before  # positive => better

        if improvement < epsilon:
            study.stop()

    return _callback


# ----------------------------------------------------------------------
#  CV objectives
# ----------------------------------------------------------------------
def make_gbts_cv_objective(
    df_raw: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[[optuna.trial.Trial], float]:
    """Factory that builds an Optuna objective for YDF-GBT.

    The returned objective:

    1. Samples a set of hyper-parameters for
       :class:`ydf.GradientBoostedTreesLearner`.
    2. Runs *stratified* K-fold CV **without leakage** using
       :func:`titanic.featurize.preprocess_fold`.
    3. Returns the *mean* validation accuracy across the ``n_splits`` folds
       (higher is better).

    Args:
        df_raw: Full Titanic training DataFrame **including** the
            ``"Survived"`` target column.
        n_splits: Number of folds for nested cross-validation. random_state:
        Seed for reproducible fold splits.

    Returns:
        An objective function suitable for ``study.optimize(objective, ...)``
        that yields ``float`` accuracy scores.

    Notes
    -----
    The narrow parameter ranges below are the result of a two-stage search.

    * **Stage 1 (exploratory)** – Swept a very wide space:

      ```python
      search_space = {
          "num_trees": (500, 5000), "shrinkage": (0.01, 0.30, "log"),
          "max_depth": (3, 12), "subsample": (0.50, 1.00),
          "num_candidate_attributes_ratio": (0.50, 1.00),
          "l2_regularization":(0.0, 3.0),
      }
      ```

    * **Stage 2 (refinement)** – Centered the bounds on the best region from
      Stage 1 to speed up convergence in CI.
    """

    def _objective(trial: optuna.trial.Trial) -> float:
        """Inner objective executed by Optuna."""
        hyperparams = {
            "num_trees": trial.suggest_int("num_trees", 260, 380),
            "shrinkage": trial.suggest_float("shrinkage", 0.14, 0.18, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 12),
            "subsample": trial.suggest_float("subsample", 0.62, 0.94),
            "num_candidate_attributes_ratio": trial.suggest_float(
                "num_candidate_attributes_ratio", 0.71, 1.0
            ),
            "min_examples": trial.suggest_int("min_examples", 2, 6),
            "l1_regularization": trial.suggest_float("l1_regularization", 0.15, 0.22),
            "l2_categorical_regularization": trial.suggest_float(
                "l2_categorical_regularization", 5.1, 7.7
            ),
            "l2_regularization": trial.suggest_float("l2_regularization", 5.8, 8.7),
            "early_stopping": "LOSS_INCREASE",
            "early_stopping_num_trees_look_ahead": 30,
        }

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        fold_acc: list[float] = []

        for tr_idx, va_idx in skf.split(df_raw, df_raw["Survived"]):
            _, _, df_tr_ydf, _, _, df_va_ydf = preprocess_fold(
                df_raw.iloc[tr_idx], df_raw.iloc[va_idx]
            )

            model = (
                ydf.GradientBoostedTreesLearner(
                    label="Survived",
                    task=ydf.Task.CLASSIFICATION,
                    **hyperparams,
                ).train(df_tr_ydf, valid=df_va_ydf)  # enables early-stopping
            )

            preds = (model.predict(df_va_ydf) >= 0.5).astype(int)
            fold_acc.append(accuracy_score(df_va_ydf["Survived"], preds))

        return float(np.mean(fold_acc))

    return _objective


def make_xgb_cv_objective(
    df_raw: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[[optuna.trial.Trial], float]:
    """Factory that builds an Optuna objective for XGBoost.

    The logic mirrors :func:`make_gbts_cv_objective` but targets
    :class:`xgboost.XGBClassifier`.  GPU acceleration is enabled automatically
    if TensorFlow can see at least one CUDA device.

    Args:
        df_raw: Titanic training set with the ``Survived`` column. n_splits:
        Number of folds for cross-validation. random_state: Seed used for
        ``StratifiedKFold``.

    Returns:
        An Optuna objective that maximizes mean CV accuracy.

    Notes
    -----
    The narrow parameter ranges below are the result of a two-stage search.

    * **Stage 1 (exploratory)** – Swept a very wide space:

      ```python
      search_space = {
          "n_estimators": (500, 5000),
          "max_depth": (3, 12),
          "eta": (0.01, 0.30, "log"),
          "gamma": (0.0, 1.0),
          "min_child_weight": (1, 10),
          "max_delta_step": (0, 10),
          "subsample": (0.50, 1.00),
          "colsample_bytree": (0.50, 1.00),
          "colsample_bylevel": (0.50, 1.00),
          "colsample_bynode": (0.50, 1.00),
          "reg_alpha": (0.0, 2.0),
          "reg_lambda": (0.0, 2.0),
          # objective, eval_metric, etc. kept fixed:
          # {"objective": "binary:logistic", "eval_metric": "logloss", ...}
      }
      ```

    * **Stage 2 (refinement)** – Centered the bounds on the best region from
      Stage 1 to speed up convergence in CI.
    """
    # Local import keeps TensorFlow optional
    import tensorflow as tf  # pylint: disable=import-error

    def _objective(trial: optuna.trial.Trial) -> float:
        """Inner objective executed by Optuna."""
        hyperparams = {
            "n_estimators": trial.suggest_int("n_estimators", 2800, 4200),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "eta": trial.suggest_float("eta", 0.10, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
            "gamma": trial.suggest_float("gamma", 0.28, 0.42),
            "subsample": trial.suggest_float("subsample", 0.51, 0.77),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.71, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.26, 0.40),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.80, 1.0),
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "early_stopping_rounds": 30,
        }

        # Automatically switch to GPU if available
        if tf.config.list_physical_devices("GPU"):
            hyperparams.update({"device": "cuda", "tree_method": "hist"})

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        fold_acc: list[float] = []

        for tr_idx, va_idx in skf.split(df_raw, df_raw["Survived"]):
            X_tr, y_tr, _, X_va, y_va, _ = preprocess_fold(
                df_raw.iloc[tr_idx], df_raw.iloc[va_idx]
            )

            model = xgb.XGBClassifier(**hyperparams)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict_proba(X_va)[:, 1] >= 0.5
            fold_acc.append(accuracy_score(y_va, preds))

        return float(np.mean(fold_acc))

    return _objective
