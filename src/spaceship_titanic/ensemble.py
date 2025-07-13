"""Ensembling utilities for the Spaceship Titanic competition.

This module provides two helper objects:

* **`run_fold`** – Trains CatBoost, LightGBM and XGBoost on one train/validation
  split and returns out-of-fold probabilities for each learner.  The routine is
  GPU-aware (early-stopping) but otherwise stateless, making it safe to call in
  parallel.

* **`weight_objective`** – Factory that returns an Optuna objective function
  used to learn optimal soft-vote weights for the three OOF probability
  matrices produced by `run_fold`.

Both helpers are imported by `train.py` and the notebooks; they contain **no
side-effects** and depend only on NumPy arrays already prepared upstream.
"""

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score

from .train import make_fold_data


def run_fold(tr_idx, va_idx, cat_params, lgb_params, xgb_params, X, y):
    """Fit three GBM learners on a single cross-validation fold.

    Args:
        tr_idx (np.ndarray): Row indices comprising the training split.
        va_idx (np.ndarray): Row indices comprising the validation split.
        cat_params (dict): Hyper-parameters for ``catboost.CatBoostClassifier``.
        lgb_params (dict): Hyper-parameters for ``lightgbm.train``.
        xgb_params (dict): Hyper-parameters for ``xgboost.XGBClassifier``.
        X (pd.DataFrame or np.ndarray): Full feature matrix
            (untransformed). The helper calls :pyfunc:`make_fold_data`
            internally to apply the feature pipeline.
        y (pd.Series or np.ndarray): Binary target vector aligned with ``X``.

    Returns:
        tuple: ``(va_idx, p_cat, p_lgb, p_xgb)`` where

        * ``va_idx`` – the original validation indices (for scatter-gather);
        * ``p_cat`` – NumPy array of CatBoost validation probabilities
          (shape = len(va_idx), dtype =float);
        * ``p_lgb`` – LightGBM validation probabilities;
        * ``p_xgb`` – XGBoost validation probabilities.

    Notes:
        * Each model is trained with **early stopping** (200 rounds) on its
          respective validation split.
        * All models are fitted using the supplied hyper-parameter dictionaries
          **without modification**; tweak them upstream if needed.
    """

    X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)

    # CatBoost
    cat = cb.CatBoostClassifier(**cat_params).fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=200,
        use_best_model=True,
    )
    p_cat = cat.predict_proba(X_va)[:, 1]

    # LightGBM
    d_tr = lgb.Dataset(X_tr, y_tr)
    d_va = lgb.Dataset(X_va, y_va)
    lgbm = lgb.train(
        lgb_params,
        d_tr,
        valid_sets=[d_va],
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    p_lgb = lgbm.predict(X_va, num_iteration=lgbm.best_iteration)

    # XGBoost
    xgbc = xgb.XGBClassifier(**xgb_params).fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=200, verbose=False
    )
    p_xgb = xgbc.predict_proba(X_va)[:, 1]

    return va_idx, p_cat, p_lgb, p_xgb


def weight_objective(oof_cat, oof_lgb, oof_xgb, y):
    """Return an Optuna objective that finds optimal soft-vote weights.

    The inner objective normalises three candidate weights so they sum to 1,
    blends the out-of-fold probability matrices, thresholds at 0.5 and scores
    the result with accuracy.

    Args:
        oof_cat (np.ndarray): OOF probabilities from CatBoost
            (shape = ``(n_samples,)``).
        oof_lgb (np.ndarray): OOF probabilities from LightGBM.
        oof_xgb (np.ndarray): OOF probabilities from XGBoost.
        y (np.ndarray or pd.Series): Ground-truth binary labels.

    Returns:
        Callable[[optuna.trial.Trial], float]: Objective function compatible
        with :pyfunc:`optuna.study.Study.optimize`.

    Example:
        >>> study = optuna.create_study(direction="maximize")
        >>> study.optimize(
        ...     weight_objective(oof_cat, oof_lgb, oof_xgb, y),
        ...     n_trials=150,
        ... )
    """

    def _objective(trial):
        w_cat = trial.suggest_float("w_cat", 0.0, 1.0)
        w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        w_sum = w_cat + w_lgb + w_xgb + 1e-12
        w_cat, w_lgb, w_xgb = w_cat / w_sum, w_lgb / w_sum, w_xgb / w_sum

        proba = w_cat * oof_cat + w_lgb * oof_lgb + w_xgb * oof_xgb
        preds = proba >= 0.5
        return accuracy_score(y, preds)

    return _objective
