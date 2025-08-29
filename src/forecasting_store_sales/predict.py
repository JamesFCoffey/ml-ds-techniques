"""
Utilities for generating one-step-ahead (typically one day) hybrid forecasts
for the Store Sales problem. The hybrid approach combines:

  1) A baseline multi-output model (e.g., Ridge) that predicts sales directly
     from deterministic and lag features.
  2) A second model (e.g., Gradient Boosted Trees) trained on the baseline
     residuals with rich categorical metadata and compact promotion features.

The main entry point, `hybrid_predict_one_day`, takes a design matrix for the
forecast horizon (usually a single date), produces baseline predictions for all
(store, family) series at once, constructs the residual feature matrix aligned
to these series, and then adds the residual model’s predictions back to form
the final sales forecast.

Typical usage in an iterative forecast loop is:

    for date in forecast_index:
        X_test = build_features_for(date)          # 1-row DataFrame
        y_next = hybrid_predict_one_day(           # wide (store,family) columns
            X_test, y_train_wide, baseline, resid_model,
            stores, local_holidays, regional_holidays, onp_long,
            BASELINE_COLS, RESIDUAL_COLS, FAMILY_CATS, STORE_CATS, TYPE_CATS
        )
        y_history = pd.concat([y_history, y_next]) # roll history forward

Notes:
  * This module assumes “wide” targets where columns are a MultiIndex
    `(store_nbr, family)` and the index is a daily PeriodIndex or DatetimeIndex.
  * Holiday tables are expected to be de-duplicated for the merge keys used
    (see `validate="many_to_one"` in the merges).
"""

import pandas as pd


def hybrid_predict_one_day(
    X_test,
    y,
    model_1,
    model_2,
    stores,
    local_holidays,
    regional_holidays,
    onp_long,
    BASELINE_COLS,
    RESIDUAL_COLS,
    FAMILY_CATS,
    STORE_CATS,
    TYPE_CATS,
):
    """Produce hybrid forecasts for one (or more) forecast dates.

    This function implements a two-stage, add-on residual approach:

    1) **Baseline prediction** (multi-output):
       Uses `model_1` to predict sales for all `(store_nbr, family)` series
       given a design matrix `X_test`. The output is a wide DataFrame with the
       same column MultiIndex as `y.columns`.

    2) **Residual enhancement**:
       Stacks the baseline predictions to long form and constructs a residual
       feature matrix by joining:
         - Store metadata (`stores`: city, state, type, cluster)
         - Local and regional holiday indicators for the matching date and
           location (city/state)
         - Compact promotion features (`onp_long`)
         - Deterministic one-hots for family/store/type/cluster built from
           fixed category sets (FAMILY_CATS, STORE_CATS, TYPE_CATS)

       The residual model (`model_2`) predicts adjustments that are added back
       to the baseline, with the final result clipped at zero.

    The function is named “one_day” because it is typically called with
    a single-row `X_test` (one forecast date). It will also work for multiple
    dates at once (multiple rows in `X_test`) provided all merges have coverage.

    Args:
      X_test (pd.DataFrame):
        Design matrix for the forecast horizon. Index should be daily
        `PeriodIndex('D')` or `DatetimeIndex`. Columns must include the exact
        features expected by `model_1`; columns are aligned using
        `BASELINE_COLS` via `reindex(..., fill_value=0)`.
      y (pd.DataFrame):
        Wide target frame used to copy the output column structure for the
        baseline prediction. Index is daily; columns are a MultiIndex
        `(store_nbr, family)`. Only `y.columns` is used.
      model_1:
        Baseline, scikit-learn–like estimator with `predict(X) -> array`
        of shape `(n_rows, n_series)`, where `n_series == len(y.columns)`.
      model_2:
        Residual learner, scikit-learn–like estimator with `predict(X) -> array`
        of length equal to the number of stacked rows in the residual feature
        matrix (i.e., `#dates × #stores × #families`).
      stores (pd.DataFrame):
        Store metadata indexed by `store_nbr`. Must contain columns
        `['city', 'state', 'type', 'cluster']`. Recommended dtypes are
        categoricals for `type` and `cluster`.
      local_holidays (pd.DataFrame):
        Local holiday table indexed by `date` with columns at least
        `['city', 'additional', 'additional_squared', 'local_holiday']`.
        Must be de-duplicated per `(date, city)` for `validate='many_to_one'`.
      regional_holidays (pd.DataFrame):
        Regional holiday table indexed by `date` with columns at least
        `['state', 'regional_holiday']`. Must be de-duplicated per `(date, state)`.
      onp_long (pd.DataFrame):
        Compact promotion features with MultiIndex index
        `('date', 'store_nbr', 'family')` and a small set of numeric columns
        (e.g., `'onp_expo_0_2'`, `'onp_expo_8_14'`, `'onp_start_lead_0'`).
      BASELINE_COLS (Sequence[str]):
        Exact feature names (and order) used to train `model_1`. `X_test` is
        reindexed to these columns; any missing features are filled with 0.0.
      RESIDUAL_COLS (Sequence[str]):
        Exact feature names (and order) used to train `model_2`. The constructed
        residual matrix is reindexed to these columns; any missing features are
        filled with 0.0.
      FAMILY_CATS (Sequence):
        Fixed category set for `family` used to create deterministic one-hot
        columns during inference (ensures train/test alignment).
      STORE_CATS (Sequence):
        Fixed category set for `store_nbr` used to create deterministic one-hot
        columns during inference.
      TYPE_CATS (Sequence):
        Fixed category set for `type` used to create deterministic one-hot
        columns during inference.

    Returns:
      pd.DataFrame:
        Wide DataFrame of predictions with index equal to `X_test.index`
        (one or more dates) and columns as a MultiIndex `(store_nbr, family)`.
        Values are non-negative floats (sales clipped at 0).

    Raises:
      pandas.errors.MergeError: If the local or regional holiday merges violate
        the `validate='many_to_one'` constraint (e.g., duplicate keys).
      KeyError: If required columns are missing from `stores`,
        `local_holidays`, `regional_holidays`, or `onp_long`.
      ValueError: If `model_1.predict` or `model_2.predict` return arrays with
        unexpected shapes that cannot be reshaped/aligned to the target index.

    Examples:
      Predict a single day:

      >>> y_pred = hybrid_predict_one_day(
      ...     X_test=features_for_day,   # 1×F DataFrame
      ...     y=y_wide,                  # train target, used for columns only
      ...     model_1=baseline_model,
      ...     model_2=residual_model,
      ...     stores=stores_df,
      ...     local_holidays=local_df,
      ...     regional_holidays=regional_df,
      ...     onp_long=onp_compact_long,
      ...     BASELINE_COLS=baseline_cols,
      ...     RESIDUAL_COLS=residual_cols,
      ...     FAMILY_CATS=family_categories,
      ...     STORE_CATS=store_categories,
      ...     TYPE_CATS=type_categories,
      ... )
      >>> y_pred.shape
      (1, len(y_wide.columns))  # columns are (store_nbr, family)

    Notes:
      * All one-hot expansions are built **without** `drop_first` and then
        aligned to `RESIDUAL_COLS`, ensuring exact train/test column parity.
      * Any missing residual features after alignment are filled with 0.0,
        which is safe as long as the training pipeline also saw explicit zeros
        for “not applicable / absent” categories and indicators.
    """

    # --- Baseline prediction (wide) ------------------------------------------
    Xb = X_test.reindex(columns=BASELINE_COLS, fill_value=0).astype("float32")
    y_pred_1 = pd.DataFrame(model_1.predict(Xb), index=Xb.index, columns=y.columns)

    # --- Build residual features X2 for this date ----------------------------
    X2 = (
        y_pred_1.stack(["store_nbr", "family"])
        .rename("sales")
        .reset_index()  # columns: date, store_nbr, family, sales
        .drop(columns=["sales"])
        .set_index("date")
    )

    # Store metadata
    X2 = X2.join(stores, on="store_nbr", how="left")

    # Local holidays
    X2 = (
        X2.reset_index()
        .merge(
            local_holidays.reset_index(),
            on=["date", "city"],
            how="left",
            validate="many_to_one",
        )
        .set_index("date")
        .drop(columns="city")
        .fillna({"additional": 0.0, "additional_squared": 0.0, "local_holiday": 0.0})
    )

    # Regional
    X2 = (
        X2.reset_index()
        .merge(
            regional_holidays.reset_index(),
            on=["date", "state"],
            how="left",
            validate="many_to_one",
        )
        .set_index("date")
        .drop(columns="state")
        .fillna({"regional_holiday": 0.0})
    )

    # Categorical domains -> deterministic one-hot
    # cast categorical domains so get_dummies sees the full space
    X2 = X2.reset_index()
    X2["family"] = X2["family"].astype(pd.CategoricalDtype(categories=FAMILY_CATS))
    X2["store_nbr"] = X2["store_nbr"].astype(pd.CategoricalDtype(categories=STORE_CATS))
    X2["type"] = X2["type"].astype(pd.CategoricalDtype(categories=TYPE_CATS))
    X2["cluster"] = X2["cluster"].astype("category")  # numeric is fine too

    # build dummies WITHOUT drop_first
    fam_oh = pd.get_dummies(X2["family"], dtype="float32")
    store_oh = pd.get_dummies(X2["store_nbr"], prefix="store_nbr_", dtype="float32")
    type_oh = pd.get_dummies(X2["type"], prefix="type_", dtype="float32")
    clus_oh = pd.get_dummies(X2["cluster"], prefix="cluster", dtype="float32")

    X2 = pd.concat(
        [
            X2.drop(columns=["type", "cluster"], errors="ignore"),
            fam_oh,
            store_oh,
            type_oh,
            clus_oh,
        ],
        axis=1,
    )

    # back to the 3-level index
    X2 = X2.set_index(["date", "store_nbr", "family"]).sort_index()

    # onpromotion compact features
    X2 = (
        X2.reset_index()
        .merge(
            onp_long.reset_index(),
            on=["date", "store_nbr", "family"],
            how="left",
            validate="many_to_one",
        )
        .set_index(["date", "store_nbr", "family"])
    )

    # reindex to the exact training columns + order
    X2 = X2.reindex(columns=RESIDUAL_COLS, fill_value=0).astype("float32")

    # --- Residual prediction and hybrid --------------------------------------
    y1_long = (
        y_pred_1.stack(["store_nbr", "family"])
        .rename("Baseline")
        .reorder_levels(["date", "store_nbr", "family"])
        .sort_index()
    )

    y2_pred = pd.Series(model_2.predict(X2), index=X2.index, name="Res2")
    y_pred_long = (y1_long.loc[y2_pred.index] + y2_pred).rename("sales").clip(lower=0)

    # Wide (columns MultiIndex like training y)
    y_pred_df = y_pred_long.unstack(["store_nbr", "family"])
    return y_pred_df
