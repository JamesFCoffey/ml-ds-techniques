"""Top-level package for the `forecasting_store_sales` toolkit.

This package provides a small, opinionated toolkit for the Kaggle
*Store Sales – Time Series Forecasting* problem (and similar retail time-series
setups). It exposes a stable, convenience API via re-exports so you can do:

    from forecasting_store_sales import make_dp, fit_ridge_with_optuna, hybrid_predict_one_day

without reaching into submodules.

The public surface groups into five themes:

* **utils** – Lightweight helpers for time-index handling and normalization.
* **features** – Feature engineering utilities (deterministic trends/seasonality,
  lags, compact onpromotion features, oil macro signals, transactions shaping).
* **modeling** – Model-building and hyperparameter tuning helpers (Optuna +
  scikit-learn pipelines).
* **plots** – Analysis/diagnostic plots (periodograms, lag plots, CCF utilities,
  event studies).
* **predict** – End-to-end daily hybrid prediction for baseline + residual models.

Example:
  ```python
  from forecasting_store_sales import make_dp, fit_ridge_with_optuna

  X_tr = make_dp(index=y_train.index, order=2, weekly=True, fourier_order=4)
  X_va = X_tr.loc[y_valid.index]
  pipe, study = fit_ridge_with_optuna(
      X_tr, y_train, X_va, y_valid, alpha_lo=1e-2, alpha_hi=1e3, patience=10, seed=123
  )
  ```

Notes:

* All re-exported symbols listed in `__all__` are considered part of the
  stable public API for this package.
* Submodules remain importable if you prefer explicit namespacing, e.g.
  `from forecasting_store_sales.features import make_dp`.

"""

from .features import (
    build_local_table,
    build_regional_table,
    create_onpromo_feat,
    fill_one_store,
    lags_row_for_date,
    make_dp,
    make_lags,
    make_oil_weekly_features,
    make_wide_target,
    make_wide_transactions,
)
from .modeling import fit_ridge_with_optuna, make_early_stopping_callback
from .plots import (
    event_study_onpromotion_starts,
    lagplot,
    plot_ccf_oil_global_long,
    plot_ccf_onpromotion_pair_prewhiten,
    plot_ccf_transactions_store,
    plot_lags,
    plot_periodogram,
    seasonal_plot,
    summarize_onpromo_ccf_by_pair_prewhiten,
    summarize_transactions_ccf_allstores,
)
from .predict import hybrid_predict_one_day
from .utils import _ensure_daily, _match_key_in_index_like, zscore_cols

__all__ = [
    # utils
    "_ensure_daily",
    "zscore_cols",
    "_match_key_in_index_like",
    # features
    "make_wide_target",
    "make_dp",
    "make_lags",
    "lags_row_for_date",
    "fill_one_store",
    "make_wide_transactions",
    "make_oil_weekly_features",
    "build_local_table",
    "build_regional_table",
    "create_onpromo_feat",
    # modeling
    "make_early_stopping_callback",
    "fit_ridge_with_optuna",
    # plots
    "seasonal_plot",
    "plot_periodogram",
    "lagplot",
    "plot_lags",
    "event_study_onpromotion_starts",
    "summarize_onpromo_ccf_by_pair_prewhiten",
    "plot_ccf_onpromotion_pair_prewhiten",
    "plot_ccf_oil_global_long",
    "summarize_transactions_ccf_allstores",
    "plot_ccf_transactions_store",
    # predict
    "hybrid_predict_one_day",
]
