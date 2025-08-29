"""
Lightweight utility helpers for time series preparation and indexing.

This module provides:

* Daily index normalization for pandas Series while preserving values.
* Column-wise z-score scaling using the population standard deviation.
* A forgiving key matcher that resolves a provided key to one of several
  candidate labels by comparing their string representations.

All functions are designed to be small, dependency-free wrappers around common
pandas/numpy idioms used throughout the Store Sales project.
"""

import numpy as np
import pandas as pd


def _ensure_daily(s: pd.Series) -> pd.Series:
    """Return a Series reindexed to a daily `DatetimeIndex` (no filling).

    The function normalizes the index to daily granularity without imputing
    values:

    * If the index is a `PeriodIndex`, it is converted to timestamps.
    * Otherwise, the index is parsed with `pd.to_datetime`.
    * The series is then reindexed to daily frequency via `.asfreq('D')`,
      which inserts missing days as NaN (no forward/backward fill).

    Args:
      s: Input series with a date-like index (e.g., `DatetimeIndex` or
        `PeriodIndex`), or an index coercible via `pd.to_datetime`.

    Returns:
      pd.Series: A copy of `s` whose index is a daily `DatetimeIndex` and
      whose missing days are represented as NaN.

    Examples:
      >>> s = pd.Series([1, 2], index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
      >>> _ensure_daily(s).index.freq
      <Day>
      >>> _ensure_daily(s).isna().sum()
      1
    """
    s = s.copy()
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        s.index = idx.to_timestamp()
    else:
        s.index = pd.to_datetime(idx)
    return s.asfreq("D")


def zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column-wise z-scoring using the population standard deviation.

    This standardizes each column independently as `(x - mean) / std`, where
    `std` is computed with `ddof=0`. Columns with zero variance are set to NaN
    (to avoid division by zero). The index and column labels are preserved.

    Args:
      df: DataFrame of numeric columns to standardize.

    Returns:
      pd.DataFrame: Z-scored copy of `df`. Constant columns become all-NaN.

    Notes:
      * Uses `df.std(ddof=0)`, i.e., the population standard deviation.
      * Non-numeric columns will propagate `NaN`/raise depending on dtypes and
        pandas version; prefer passing numeric columns only.

    Examples:
      >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
      >>> z = zscore_cols(df)
      >>> np.isfinite(z["a"]).all(), z["b"].isna().all()
      (True, True)
    """
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - mu) / sd


def _match_key_in_index_like(key, candidates):
    """Resolve `key` to one of `candidates` by string equivalence.

    This helper builds a mapping from `str(candidate)` to the original candidate
    object and returns the candidate whose string form equals `str(key)`.
    Useful when user input (e.g., `"1"` or `'('A', 3)'`) must match labels that
    may be ints, tuples, categoricals, etc.

    Args:
      key: The lookup key (any type); compared via `str(key)`.
      candidates: Iterable of candidate labels/keys.

    Returns:
      Any: The original candidate object that matches `key` by `str()`.

    Raises:
      KeyError: If no candidate matches `str(key)`. The error message includes
        the first few available keys to aid debugging.

    Examples:
      >>> candidates = [1, 2, 10]
      >>> _match_key_in_index_like("2", candidates)
      2
      >>> _match_key_in_index_like("3", candidates)
      Traceback (most recent call last):
        ...
      KeyError: "'3' not found. Available (first 10): ['1', '2', '10']"
    """
    m = {str(c): c for c in candidates}
    if str(key) not in m:
        raise KeyError(
            f"{key!r} not found. Available (first 10): {list(m.keys())[:10]}"
        )
    return m[str(key)]
