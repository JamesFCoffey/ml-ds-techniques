"""
Optuna utilities: early-stopping callback and CV objectives
for YDF-GBT and XGBoost models.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import ydf
from optuna.study import StudyDirection

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .featurize import preprocess_fold
from .models import YDFWrapper


# ----------------------------------------------------------------------
#  Generic convergence / pruning callback
# ----------------------------------------------------------------------
def make_convergence_callback(
    start_trial: int = 200,
    lookback:    int = 100,
    epsilon:     float = 1e-4,
):
    """
    Generic early-stop callback:
      • MINIMIZE  → stop if best loss hasn’t dropped by ≥epsilon
      • MAXIMIZE  → stop if best score hasn’t risen  by ≥epsilon
    """
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        finished = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        n_completed = len(finished)
        if n_completed < max(start_trial, lookback + 1):
            return                                    # not enough data yet

        # Sort by trial number (ascending)
        finished.sort(key=lambda t: t.number)

        if study.direction == StudyDirection.MINIMIZE:
            best_before = min(t.value for t in finished[:-lookback])
            best_now    = min(t.value for t in finished)
            improvement = best_before - best_now      # positive if better
        else:  # MAXIMIZE
            best_before = max(t.value for t in finished[:-lookback])
            best_now    = max(t.value for t in finished)
            improvement = best_now - best_before      # positive if better

        if improvement < epsilon:
            study.stop()

    return _callback

# ----------------------------------------------------------------------
#  CV objectives
# ----------------------------------------------------------------------
def make_gbts_cv_objective(
    df_raw: pd.DataFrame, n_splits: int = 5, random_state: int = 42
):
    """
    Returns an Optuna objective that runs StratifiedKFold CV on df_raw
    (which must include 'Survived') and returns the mean accuracy.
    """

    def _objective(trial):
        # ORIGINAL HYPERPARAMETER SEARCH SPACE:
        # "num_trees": trial.suggest_int("num_trees", 500, 5000),
        # "shrinkage": trial.suggest_float("shrinkage", 0.01, 0.3, log=True),
        # "max_depth": trial.suggest_int("max_depth", 3, 12),
        # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # "num_candidate_attributes_ratio": trial.suggest_float(
        #     "num_candidate_attributes_ratio", 0.5, 1.0
        # ),
        # "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 3.0),
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

        # Stratified K-fold on the 'Survived' column
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        fold_acc = []

        for tr_idx, va_idx in skf.split(df_raw, df_raw["Survived"]):
            _, _, df_tr_ydf, \
            _, _, df_va_ydf = preprocess_fold(df_raw.iloc[tr_idx],
                                              df_raw.iloc[va_idx])

            model = ydf.GradientBoostedTreesLearner(
                label="Survived",
                task=ydf.Task.CLASSIFICATION,
                **hyperparams
            ).train(
                df_tr_ydf,
                valid=df_va_ydf,  # enables early stopping
            )
            
            # Accuracy on validation
            preds = (model.predict(df_va_ydf) >= 0.5).astype(int)
            fold_acc.append(accuracy_score(df_va_ydf["Survived"], preds))

        # Return mean fold accuracy
        return float(np.mean(fold_acc))

    return _objective

def make_xgb_cv_objective(
    df_raw: pd.DataFrame, n_splits: int = 5, random_state: int = 42
):
    """
    Returns an Optuna objective that runs StratifiedKFold CV on df_raw
    and returns the mean accuracy across folds.
    """

    import tensorflow as tf  # local import to avoid mandatory TF dependency

    def _objective(trial):
        # ORIGINAL HYPERPARAMETER SEARCH SPACE:
        # "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
        # "max_depth": trial.suggest_int("max_depth", 3, 12),
        # "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        # "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        # "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        # "max_delta_step":   trial.suggest_int("max_delta_step",   0,  10),
        # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        # "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.5, 1.0),
        # "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        # "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        # "objective": "binary:logistic",
        # "use_label_encoder": False,
        # "eval_metric": "logloss",
        # "random_state": 42,
        # "verbosity": 0,
        # "early_stopping_rounds": 30,
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

        # GPU auto‐detection
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            hyperparams.update({"device": "cuda", "tree_method": "hist"})

        # Stratified K-fold
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        fold_acc = []
        
        for tr_idx, va_idx in skf.split(df_raw, df_raw["Survived"]):
            X_tr, y_tr, _, \
            X_va, y_va, _ = preprocess_fold(df_raw.iloc[tr_idx],
                                            df_raw.iloc[va_idx])

            model = xgb.XGBClassifier(**hyperparams)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            fold_acc.append(
                accuracy_score(y_va, (model.predict_proba(X_va)[:,1] >= 0.5))
            )

        # All folds completed → return the final mean accuracy
        return float(np.mean(fold_acc))

    return _objective