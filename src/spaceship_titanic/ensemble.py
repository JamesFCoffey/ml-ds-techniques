import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score

from .train import make_fold_data


def run_fold(tr_idx, va_idx, cat_params, lgb_params, xgb_params, X, y):
    X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)

    # CatBoost (early-stop)
    cat = cb.CatBoostClassifier(**cat_params).fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=200,
        use_best_model=True,
    )
    p_cat = cat.predict_proba(X_va)[:, 1]

    # LightGBM (early-stop)
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

    # XGBoost (early-stop)
    xgbc = xgb.XGBClassifier(**xgb_params).fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=200, verbose=False
    )
    p_xgb = xgbc.predict_proba(X_va)[:, 1]

    return va_idx, p_cat, p_lgb, p_xgb


def weight_objective(oof_cat, oof_lgb, oof_xgb, y):
    def _objective(trial):
        w_cat = trial.suggest_float("w_cat", 0.0, 1.0)
        w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        w_sum = w_cat + w_lgb + w_xgb + 1e-12
        w_cat, w_lgb, w_xgb = w_cat / w_sum, w_lgb / w_sum, w_xgb / w_sum

        proba = w_cat * oof_cat + w_lgb * oof_lgb + w_xgb * oof_xgb
        preds = proba >= 0.5  # decision threshold tuned later if desired
        return accuracy_score(y, preds)

    return _objective
