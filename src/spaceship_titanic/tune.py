"""Hyper-parameter optimization objectives for gradient-boosting models.

This module wraps the Optuna search spaces for the three tree boosters used in
the Spaceship Titanic project:

* **CatBoost** – handles categorical features on-device and offers GPU training.
* **LightGBM** – histogram-based GBT with efficient GPU implementation.
* **XGBoost** – reference boosted tree library, GPU or CPU.

Each public helper returns a *closure* that Optuna can pass to
`Study.optimize()`.  That closure trains the model inside every fold of a
supplied `cv` splitter, scores mean accuracy, and lets Optuna maximize the
result.  Keeping the factories here decouples notebook experimentation from the
core training logic.

"""

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from .train import make_fold_data


def objective_cat(cv, X, y, groups, gpus):
    """Create a CatBoost Optuna objective.

    Args:
        cv (BaseCrossValidator): Group-aware cross-validator that yields
            `(train_idx, val_idx)` pairs.
        X (pd.DataFrame): Feature matrix **before** transformation.
        y (pd.Series | np.ndarray): Binary target aligned with ``X``.
        groups (pd.Series | np.ndarray): Group labels used by ``cv`` to avoid
            leakage.
        gpus (bool): If *True*, enable GPU training.

    Returns:
        Callable[[optuna.trial.Trial], float]: Objective that Optuna can
        maximise. The objective trains CatBoost with early stopping on each
        fold and returns the **mean fold accuracy**.

    Notes:
        The search space tunes depth, learning rate, regularization, bagging
        temperature and border count while capping iterations at 4 000 with
        early stopping after 50 rounds.

    """

    def _objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
            "border_count": trial.suggest_int("borders", 32, 255),
            "iterations": 4000,  # upper bound – early-stop
            "random_state": 42,
            "verbose": False,
        }
        if gpus:
            params["task_type"] = "GPU"

        accs = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)
            model = cb.CatBoostClassifier(**params).fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
            )
            preds = model.predict_proba(X_va)[:, 1] >= 0.5
            accs.append(accuracy_score(y_va, preds))
        return np.mean(accs)

    return _objective


def objective_lgb(cv, X, y, groups):
    """Create a LightGBM Optuna objective.

    Args:
        cv (BaseCrossValidator): Cross-validator that respects group structure.
        X (pd.DataFrame): Raw features.
        y (pd.Series | np.ndarray): Binary labels.
        groups (pd.Series | np.ndarray): Group identifiers for CV splitting.

    Returns:
        Callable[[optuna.trial.Trial], float]: Objective that trains a
        LightGBM model in each fold (early stopping 50 rounds) and returns the
        mean validation accuracy.

    Search Space:
        * ``learning_rate``  – log-uniform 0.01-0.2
        * ``num_leaves``     – 15-63
        * ``min_data_in_leaf``
        * ``feature_fraction`` / ``bagging_fraction`` / ``bagging_freq``
        * ``lambda_l1`` / ``lambda_l2``

    GPU training is forced via ``device_type='gpu'`` for speed.

    """

    def _objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting": "gbdt",
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("leaves", 15, 63),
            "min_data_in_leaf": trial.suggest_int("min_leaf", 5, 40),
            "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bfreq", 1, 7),
            "lambda_l1": trial.suggest_float("l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("l2", 0.0, 5.0),
            "max_depth": -1,
            "verbosity": -1,
            "force_col_wise": True,
            "device_type": "gpu",
        }

        accs = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)
            d_train = lgb.Dataset(X_tr, y_tr)
            d_valid = lgb.Dataset(X_va, y_va)

            mdl = lgb.train(
                params,
                d_train,
                num_boost_round=3000,
                valid_sets=[d_valid],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = mdl.predict(X_va, num_iteration=mdl.best_iteration) >= 0.5
            accs.append(accuracy_score(y_va, preds))
        return np.mean(accs)

    return _objective


def objective_xgb(cv, X, y, groups, gpus):
    """Create an XGBoost Optuna objective.

    Args:
        cv (BaseCrossValidator): Group-aware splitter.
        X (pd.DataFrame): Feature frame.
        y (pd.Series | np.ndarray): Target vector.
        groups (pd.Series | np.ndarray): Group labels to keep folds disjoint.
        gpus (bool): Enable GPU histogram algorithm when *True*.

    Returns:
        Callable[[optuna.trial.Trial], float]: Objective function that
        early-stops after 40 rounds and reports mean accuracy across folds.

    Tuned Hyper-parameters:
        ``learning_rate``, ``max_depth``, ``min_child_weight``, ``gamma``,
        ``subsample``, ``colsample_bytree``, ``reg_alpha``, ``reg_lambda``,
        and ``n_estimators`` (800-2500).

    """

    def _objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("lambda", 0.0, 5.0),
            "n_estimators": trial.suggest_int("estimators", 800, 2500),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        if gpus:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        else:
            params["tree_method"] = "hist"

        accs = []
        for tr_idx, va_idx in cv.split(X, y, groups):
            X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)
            mdl = xgb.XGBClassifier(**params).fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=40,
                verbose=False,
            )
            preds = mdl.predict_proba(X_va)[:, 1] >= 0.5
            accs.append(accuracy_score(y_va, preds))
        return np.mean(accs)

    return _objective
