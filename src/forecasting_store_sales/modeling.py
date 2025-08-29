"""
Hyperparameter tuning and model-building utilities for the Store Sales project.

This module contains:

* An Optuna early-stopping callback factory that halts a study after a
  configurable number of non-improving trials.
* A convenience function to tune and fit a Ridge regression inside a scikit-learn
  pipeline with sensible preprocessing for mixed feature sets (numeric features
  are imputed and standardized; weekly one-hot dummies and event flags are
  passed through).

Both functions are framework-agnostic beyond their dependencies on Optuna and
scikit-learn and are intended to be reused across notebooks and scripts.
"""

import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_early_stopping_callback(patience=3, min_delta=0.0):
    """Create an Optuna callback that stops after `patience` non-improving trials.

    The callback compares the study's current best objective value after each
    trial with the previously recorded best (stored in `study.user_attrs`).
    If the improvement is smaller than `min_delta` for `patience` consecutive
    trials, the study is stopped early via `study.stop()`.

    The following attributes are stored in `study.user_attrs`:

    * ``prev_best_value``: Float of the best value seen so far.
    * ``no_improve_count``: Integer count of consecutive non-improving trials.

    Args:
      patience: Number of consecutive trials allowed without sufficient
        improvement before stopping the study.
      min_delta: Minimum reduction in the objective value that qualifies as an
        improvement.

    Returns:
      Callable: A function with signature ``callback(study, trial)`` suitable
      for passing to ``optuna.study.Study.optimize(..., callbacks=[...])``.

    Example:
      >>> study = optuna.create_study(direction="minimize")
      >>> cb = make_early_stopping_callback(patience=5, min_delta=1e-4)
      >>> study.optimize(objective, n_trials=100, callbacks=[cb])
    """

    def _callback(study, trial):
        # Best value after this trial:
        best = study.best_value

        # What was the best before this trial?
        prev_best = study.user_attrs.get("prev_best_value", float("inf"))
        no_improve = study.user_attrs.get("no_improve_count", 0)

        # Check improvement
        if best < prev_best - min_delta:
            study.set_user_attr("prev_best_value", best)
            study.set_user_attr("no_improve_count", 0)
        else:
            no_improve += 1
            study.set_user_attr("no_improve_count", no_improve)
            if no_improve >= patience:
                print(
                    f"[Early stop] No improvement in {patience} consecutive trials. "
                    f"Best RMSLE so far: {best:.6f}"
                )
                study.stop()

    return _callback


def fit_ridge_with_optuna(
    X_train, y_train, X_valid, y_valid, *, alpha_lo, alpha_hi, patience=12, seed=42
):
    """Tune and fit a Ridge regression with preprocessing via Optuna.

    This function builds a scikit-learn `Pipeline` composed of:

    * A `ColumnTransformer` that:
      - Imputes (median) and standardizes numeric-like features (columns whose
        names contain one of: ``trend``, ``trend_squared``, ``sin(``, ``cos(``,
        ``y_lag_``, ``tx_lag_``, ``oil_``, ``additional``, ``days_from_``).
      - Passes through weekly dummy columns (names starting with ``s(``).
      - Passes through any remaining "event" columns.
    * A `Ridge` estimator with `fit_intercept=True`.

    The Optuna objective minimizes RMSLE on the provided validation set. To make
    RMSLE well-defined, predictions are clipped at 0 and `y_valid` is clipped
    at a lower bound of 0. For convenience, each trial also records the RMSE
    computed on the validation set in `trial.user_attrs["rmse"]`.

    After optimization, the best alpha is refit on the training data and the
    fitted pipeline together with the Optuna study are returned.

    Args:
      X_train: Training features as a pandas DataFrame. Column names determine
        which preprocessing branch each feature is routed to (numeric/dummy/event).
      y_train: Training target. Can be a 1D Series/array or a 2D DataFrame/array
        for multi-output regression (Ridge supports multi-target).
      X_valid: Validation features DataFrame aligned to `X_train` columns.
      y_valid: Validation target aligned to `X_valid`.
      alpha_lo: Lower bound (inclusive) for the log-uniform search of `alpha`.
      alpha_hi: Upper bound (inclusive) for the log-uniform search of `alpha`.
      patience: Early-stopping patience (number of consecutive non-improving
        trials) passed to :func:`make_early_stopping_callback`.
      seed: Random seed for the Optuna sampler and the Ridge estimator.

    Returns:
      Tuple[Pipeline, optuna.study.Study]: A tuple ``(best_pipe, study)`` where
      `best_pipe` is the fitted scikit-learn Pipeline and `study` is the
      completed Optuna Study containing trial history and best parameters.

    Raises:
      ValueError: If feature column names referenced by the preprocessing
        branches are not present in `X_train`/`X_valid`.

    Notes:
      * The search space for `alpha` is sampled with
        ``trial.suggest_float(..., log=True)`` over ``[alpha_lo, alpha_hi]``.
      * The objective minimized is
        ``sqrt(mean_squared_log_error(y_valid_clipped, pred_clipped))``.
      * Dummies are passed through unscaled by default; switch to ``num_pipe``
        if your workflow benefits from scaling them.

    Example:
      >>> best_pipe, study = fit_ridge_with_optuna(
      ...     X_train, y_train, X_valid, y_valid,
      ...     alpha_lo=1e-2, alpha_hi=1e3, patience=10, seed=123
      ... )
      >>> study.best_params["alpha"]
      0.42
      >>> y_hat = best_pipe.predict(X_valid)
    """
    substrings = [
        "trend",
        "trend_squared",
        "sin(",
        "cos(",
        "y_lag_",
        "tx_lag_",
        "oil_",
        "additional",
        "days_from_",
    ]
    num_cols = [c for c in X_train.columns if any(s in c for s in substrings)]
    dum_cols = [c for c in X_train.columns if c.startswith("s(")]  # weekly one-hots
    evt_cols = [c for c in X_train.columns if c not in num_cols and c not in dum_cols]

    def objective(trial):
        alpha = trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True)
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

        prep = ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                (
                    "dum",
                    "passthrough",
                    dum_cols,
                ),  # or use num_pipe if you prefer scaling dummies
                ("evt", "passthrough", evt_cols),
            ],
            remainder="drop",
        )

        pipe = Pipeline(
            [
                ("prep", prep),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=seed)),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
        rmsle = (
            mean_squared_log_error(y_valid.clip(lower=0), np.clip(pred, 0, None)) ** 0.5
        )
        trial.set_user_attr(
            "rmse", float(mean_squared_error(y_valid, pred, squared=False))
        )
        return rmsle

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        objective,
        n_trials=40,
        callbacks=[make_early_stopping_callback(patience=patience, min_delta=0.0)],
    )
    best_alpha = study.best_params["alpha"]
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    prep = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            (
                "dum",
                "passthrough",
                dum_cols,
            ),  # or use num_pipe if you prefer scaling dummies
            ("evt", "passthrough", evt_cols),
        ],
        remainder="drop",
    )

    best_pipe = Pipeline(
        [
            ("prep", prep),
            ("ridge", Ridge(alpha=best_alpha, fit_intercept=True, random_state=seed)),
        ]
    ).fit(X_train, y_train)
    return best_pipe, study
