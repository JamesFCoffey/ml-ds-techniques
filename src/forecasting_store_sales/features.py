"""
Feature engineering utilities for the Store Sales project.

This module provides helpers to:

* Build the wide target matrix (date × (store_nbr, family)).
* Construct deterministic design matrices (trend, seasonality, Fourier terms)
  via `statsmodels.tsa.deterministic.DeterministicProcess`, with optional
  out-of-sample/future rows for forecasting.
* Create autoregressive lag feature blocks for wide series.
* Impute per-store daily series by filling interior gaps with day-of-week
  medians and using zeros outside the observed "open" window.
* Generate compact, long-lag oil features (weekly levels and returns) and
  align/forward-fill them to a daily index.
* Prepare deduplicated local/regional holiday lookup tables for merging.
* Build compact on-promotion features (two exposure windows + a start flag).
* Convert transactions to a wide, daily (Period[D]) matrix.
* Construct a single-row lag feature frame for a specific date, matching the
  training-time column names/order.

All functions return pandas objects and use simple, explicit conventions to
avoid data leakage (e.g., weekly features are forward-filled within a week).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from .utils import _ensure_daily


def make_wide_target(store_sales_df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide target matrix (date × (store_nbr, family)) from long data.

    Expects a long-form DataFrame with columns `["sales", "store_nbr", "family"]`
    and a date index (or a `"date"` index level). Returns a wide matrix where
    the index is date and the columns form a MultiIndex `(store_nbr, family)`.

    Args:
      store_sales_df: Long-form sales DataFrame containing a `"sales"` column
        and index/levels `"store_nbr"` and `"family"`.

    Returns:
      pd.DataFrame: Wide target matrix with date index and MultiIndex columns
      `(store_nbr, family)`, sorted by date.
    """
    return store_sales_df["sales"].unstack(["store_nbr", "family"]).sort_index()


def make_dp(
    index, *, order=2, weekly=False, freq="A", fourier_order=None, forecast=None
) -> pd.DataFrame:
    """Build a deterministic design matrix (trend/seasonality/Fourier terms).

    Uses `statsmodels.tsa.deterministic.DeterministicProcess` to create a
    regressor matrix with polynomial trend of degree `order`, optional weekly
    seasonal dummies, and optional Fourier terms at the calendar frequency
    specified by `freq`.

    If `forecast` is provided (an integer number of future steps), the function
    returns a tuple `(X_in, X_out)` where `X_out` contains out-of-sample rows.

    Args:
      index: Date-like index (e.g., `pd.DatetimeIndex` or `pd.PeriodIndex`)
        used by the deterministic process.
      order: Polynomial trend order (e.g., 1 for linear, 2 for quadratic).
      weekly: If True, include weekly seasonal dummies.
      freq: Calendar frequency for Fourier terms (e.g., `"A"`, `"Q"`, `"M"`).
      fourier_order: If provided, include `CalendarFourier(freq, order=fourier_order)`.
      forecast: If provided (int), also return `dp.out_of_sample(steps=forecast)`.

    Returns:
      pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        * If `forecast` is falsy: the in-sample design matrix.
        * If `forecast` is an int: `(X_in_sample, X_out_of_sample)`.

    Notes:
      The design matrix is created with `drop=True` so collinear columns are
      dropped automatically.
    """
    add = [CalendarFourier(freq=freq, order=fourier_order)] if fourier_order else []
    dp = DeterministicProcess(
        index=index,
        constant=False,
        order=order,
        seasonal=weekly,
        additional_terms=add,
        drop=True,
    )
    if forecast:
        return dp.in_sample(), dp.out_of_sample(steps=forecast)
    return dp.in_sample()


def make_lags(ts, lags, lead_time=1, name="y"):
    """Construct a block of lag features from a (wide or long) time series.

    For each integer `k` in `range(lead_time, lead_time + lags)`, this creates
    a column named `f"{name}_lag_{k}"` equal to `ts.shift(k)`.

    Args:
      ts: A pandas Series or DataFrame; `shift` must be defined for it.
      lags: Number of lag columns to create.
      lead_time: The first lag to include (e.g., `1` creates `y_lag_1`).
      name: Base prefix used in the generated column names.

    Returns:
      pd.DataFrame: Concatenation of shifted copies along columns, one column
      per lag `k` in `[lead_time, lead_time + lags)`.

    Examples:
      If `lags=3` and `lead_time=1`, columns will be `{name}_lag_1, 2, 3`.
    """
    return pd.concat(
        {f"{name}_lag_{i}": ts.shift(i) for i in range(lead_time, lags + lead_time)},
        axis=1,
    )


def fill_one_store(col: pd.Series, dow: pd.Index) -> pd.Series:
    """Impute a per-store daily series with DOW medians inside the open window.

    Strategy:
      * Detect the "open" window as the span from the first non-null to the
        last non-null observation.
      * Fill interior gaps within the open window using per-day-of-week medians
        computed from the observed values.
      * If any interior NaNs remain, fill them with the store-wide median.
      * Outside the open window (pre-open/post-close), fill with zeros.

    Args:
      col: Daily series for a single store (may contain NaNs).
      dow: An index-like object aligned to `col` indicating day-of-week groups
        (e.g., `col.index.dayofweek` or a categorical index of 0..6).

    Returns:
      pd.Series: A filled series with the same index as `col`.

    Notes:
      This function does **not** alter values that are already observed; it
      only fills gaps.
    """
    # If the store has no data at all, use zeros
    if not col.notna().any():
        return col.fillna(0.0)

    # Detect the "open" window: from first non-null to last non-null
    first_to_end = col.notna().cummax()
    last_to_start = col[::-1].notna().cummax()[::-1]
    open_mask = first_to_end & last_to_start
    filled = col.copy()
    na = filled.isna()

    # Per-day-of-week medians from observed data
    med_by_dow = col.groupby(dow).transform("median")

    # Internal gaps -> DOW median
    interior = open_mask & na
    filled.loc[interior] = med_by_dow.loc[interior]

    # If still NaN inside open window, use store-wide median
    still_na_inside = open_mask & filled.isna()
    if still_na_inside.any():
        filled.loc[still_na_inside] = col.median()

    # Outside open window (pre-open / post-close) -> 0
    outside = ~open_mask & filled.isna()
    if outside.any():
        filled.loc[outside] = 0.0
    return filled


def make_oil_weekly_features(
    oil_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    lag_weeks=(78, 80, 84),
    include_levels=True,
    include_returns=True,
    resample_rule="W-SUN",
    prefix="oil",
) -> pd.DataFrame:
    """Create long-lag weekly oil features and align them to a daily index.

    Pipeline:
      1) Convert oil prices to a daily series and forward-fill missing days.
      2) Aggregate to weekly means labeled at period end (`resample_rule`).
      3) Optionally compute weekly log-returns.
      4) Build lagged features at the specified week lags.
      5) Reindex to `daily_index` using forward-fill (no look-ahead within a week).

    Args:
      oil_df: DataFrame with a `"dcoilwtico"` column of price levels.
      daily_index: Target daily `DatetimeIndex` for the returned features.
      lag_weeks: Iterable of positive integers indicating the weekly lags to use.
      include_levels: If True, include lagged weekly levels.
      include_returns: If True, include lagged weekly log-returns.
      resample_rule: Pandas weekly resample rule (e.g., `"W-SUN"`).
      prefix: Feature name prefix (e.g., `"oil"`).

    Returns:
      pd.DataFrame: Columns like `"{prefix}_lvl_wlag_{k}"` and/or
      `"{prefix}_lr_wlag_{k}"`, indexed by `daily_index` (dtype float32).

    Notes:
      A positive `k` implies the driver "leads" by `k` weeks (we use `shift(k)`).
    """
    # Daily aligned price, forward-fill only
    oil_d = _ensure_daily(oil_df["dcoilwtico"].astype("float32")).ffill()

    # Weekly means labeled at week end; will be carried forward to next week safely
    oil_w = oil_d.resample(resample_rule, label="right", closed="right").mean()

    feats = {}

    if include_returns:
        # Log-returns at weekly freq
        oil_lr_w = np.log(oil_w.replace(0.0, np.nan)).ffill().diff()
        for k in lag_weeks:
            s = oil_lr_w.shift(k)  # oil leads by k weeks -> use lagged returns
            feats[f"{prefix}_lr_wlag_{k}"] = s.reindex(daily_index, method="ffill")

    if include_levels:
        for k in lag_weeks:
            s = oil_w.shift(k)  # weekly level lagged by k weeks
            feats[f"{prefix}_lvl_wlag_{k}"] = s.reindex(daily_index, method="ffill")

    out = pd.DataFrame(feats, index=daily_index).astype("float32")
    return out


def build_local_table(holidays_df):
    """Prepare a deduplicated local-holiday lookup table keyed by (date, city).

    Keeps the row with the smallest absolute `"additional"` offset when multiple
    rows exist for the same `(date, city)`. Adds a `"local_holiday" = 1.0` flag.

    Args:
      holidays_df: Holidays DataFrame with columns `["description", "locale_name",
        "additional", "additional_squared"]` and a date index (or level).

    Returns:
      pd.DataFrame: Index is `date`; columns are
        `["city", "additional", "additional_squared", "local_holiday"]`.
    """
    df = (
        holidays_df.loc[
            holidays_df["description"] == "Local",
            ["locale_name", "additional", "additional_squared"],
        ]
        .rename(columns={"locale_name": "city"})
        .reset_index()
    )
    # Prefer the closest offset (min |additional|) if duplicates on (date, city)
    df["abs_add"] = df["additional"].abs()
    df = (
        df.sort_values(["date", "city", "abs_add"])
        .drop_duplicates(["date", "city"], keep="first")
        .drop(columns="abs_add")
        .set_index("date")
    )
    df["local_holiday"] = 1.0
    return df


def build_regional_table(holidays_df):
    """Prepare a deduplicated regional-holiday table keyed by (date, state).

    Adds a `"regional_holiday" = 1.0` flag and removes duplicates per `(date, state)`.

    Args:
      holidays_df: Holidays DataFrame with columns `["description", "locale_name"]`
        and a date index (or level).

    Returns:
      pd.DataFrame: Index is `date`; columns are `["state", "regional_holiday"]`.
    """
    df = (
        holidays_df.loc[holidays_df["description"] == "Regional", ["locale_name"]]
        .rename(columns={"locale_name": "state"})
        .reset_index()
    )
    df = df.drop_duplicates(["date", "state"]).set_index("date")
    df["regional_holiday"] = 1.0
    return df


def create_onpromo_feat(df):
    """Create compact on-promotion features (exposure windows + start flag).

    Input is a long-form frame with an `"onpromotion"` column and an index that
    includes levels `("date", "store_nbr", "family")`. The function:

      * Builds a wide `(date × (store_nbr, family))` binary indicator.
      * Constructs exposure-window counts by summing future days where
        `onpromotion == 1` over two windows: `[0,2]` and `[8,14]` days ahead.
      * Creates a start flag for 0→1 transitions (`onp_start_lead_0`).
      * Stacks back to long form with columns named as strings.

    Args:
      df: Long-form DataFrame with `"onpromotion"` and index levels
        `("date", "store_nbr", "family")`.

    Returns:
      pd.DataFrame: Long-form on-promotion features indexed by
      `("date", "store_nbr", "family")` with columns:
      `["onp_expo_0_2", "onp_expo_8_14", "onp_start_lead_0"]` (float32).

    Raises:
      KeyError: If required columns or index levels are missing.
    """
    onp_wide = (
        df["onpromotion"]
        .unstack(["store_nbr", "family"])
        .sort_index()
        .astype("float32")
        .fillna(0.0)
    )
    onp_bin = (onp_wide > 0).astype("float32")

    def _sum_lead_window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        acc = None
        for k in range(start, end + 1):
            part = df.shift(-k)
            acc = part if acc is None else acc.add(part, fill_value=0.0)
        return acc

    # Two exposure windows based on your event study (+2 bump, +10 peak)
    expo_specs = {
        "onp_expo_0_2": (0, 2),
        "onp_expo_8_14": (8, 14),
    }
    onp_expo = pd.concat(
        {name: _sum_lead_window(onp_bin, a, b) for name, (a, b) in expo_specs.items()},
        axis=1,
    ).astype("float32")

    # Start flag (0→1) at t (and optionally t+1 if you like)
    onp_start = (onp_bin.astype("int8").diff() == 1).astype("float32")
    onp_start = pd.concat({"onp_start_lead_0": onp_start.shift(0)}, axis=1)

    # Assemble and convert wide→long (index: date, store_nbr, family; columns: 3 features)
    onp_feats_wide = pd.concat([onp_expo, onp_start], axis=1)
    onp_long = onp_feats_wide.stack(["store_nbr", "family"])
    onp_long.columns = onp_long.columns.map(str)  # plain strings

    return onp_long


def make_wide_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Convert transactions to a wide, daily Period[D] matrix (date × store_nbr).

    Accepts either:
      * Long format: index has levels `("date", "store_nbr")` and there is a
        `"transactions"` column, or
      * Wide format: date index with columns = store numbers. If a top-level
        MultiIndex `("transactions", store_nbr)` is present, it is sliced.

    The output index is converted to `PeriodIndex("D")` and values are cast to
    float32.

    Args:
      transactions_df: Transactions data in long or wide format (see above).

    Returns:
      pd.DataFrame: Wide matrix with `Period[D]` index and store numbers as
      columns (float32).
    """
    df = transactions_df

    # CASE A: long form in the index
    if isinstance(df.index, pd.MultiIndex) and {"date", "store_nbr"} <= set(
        df.index.names or []
    ):
        if isinstance(df, pd.DataFrame) and "transactions" in df.columns:
            ser = df["transactions"]
        else:
            ser = df.squeeze()
        y_tx = ser.unstack("store_nbr").sort_index()

    else:
        # CASE B: already wide
        y_tx = df.copy()
        # If columns are MultiIndex like ('transactions', store_nbr), slice level 0
        if (
            isinstance(y_tx.columns, pd.MultiIndex)
            and (y_tx.columns.names or [None])[0] == "transactions"
        ):
            y_tx = y_tx.xs("transactions", axis=1, level=0)

    # Ensure daily Period index
    if not isinstance(y_tx.index, pd.PeriodIndex):
        y_tx.index = pd.to_datetime(y_tx.index).to_period("D")

    return y_tx.astype("float32")


def lags_row_for_date(
    wide_df: pd.DataFrame, date, lags: int, name: str, log1p: bool = False
) -> pd.DataFrame:
    """Create a single-row lag feature frame for one target date.

    This mirrors training-time lag construction by:
      1) Ensuring `date` exists in the history,
      2) Building the full set of lags on the entire history using
         :func:`make_lags` with the same `name` and `lags`,
      3) Flattening the resulting (possibly hierarchical) column index to plain
         strings, and
      4) Returning only the row for `date`. Optionally `log1p`-transform it.

    Args:
      wide_df: Wide historical DataFrame (e.g., `date × series`) to lag.
      date: The date (compatible with the index dtype) for which to return the
        lag row.
      lags: Number of lag columns to build (starting at lag 1).
      name: Base prefix for the lag column names (e.g., `"y"`).
      log1p: If True, apply `np.log1p` to the returned single-row frame.

    Returns:
      pd.DataFrame: A one-row DataFrame indexed by `[date]`, with columns named
      `f"{name}_lag_{k}"` for `k=1..lags` (flattened), suitable for alignment
      with training columns.
    """
    # Make tmp copy of dataframe
    df = wide_df.copy()

    # Add date to index
    if date not in df.index:
        df.loc[date] = np.nan

    # build full set of lags on the *whole* history and take only the last row
    lagged = make_lags(df, lags=lags, name=name)

    # flatten to strings identical to training
    lagged.columns = lagged.columns.to_flat_index()

    # Apply optional transform
    if log1p:
        return np.log1p(lagged.loc[[date]])

    return lagged.loc[[date]]
