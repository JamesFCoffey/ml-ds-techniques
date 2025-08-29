"""
Visualization utilities for exploratory and diagnostic time-series analysis in
the Store Sales project. The helpers in this module cover:

* Seasonal line plots that facet by a calendar period (e.g., year) and plot on
  a within-period axis (e.g., day of year).
* Spectral diagnostics via a (log-x) periodogram.
* Lag plots and small-multiples of lag plots to visualize serial dependence.
* Cross-correlation function (CCF) plots between key drivers (e.g., oil prices)
  and sales residuals, including per-store summaries.
* Promotion impact diagnostics:
  - Prewhitened CCFs between on-promotion signals and residuals.
  - Event-study plots around promotion starts.

Most functions accept/return standard pandas and matplotlib objects so they can
be composed in notebooks or scripts. This module also relies on small utilities
from ``.utils`` for daily index alignment and fuzzy key matching.
"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_ccf
from statsmodels.tsa.ar_model import AutoReg

from .utils import _ensure_daily, _match_key_in_index_like


def seasonal_plot(X, y, period, freq, ax=None):
    """Draw a seasonal line plot grouped by a period and indexed by a frequency.

    For each unique value in ``X[period]`` (e.g., year), this plots a line of
    ``y`` against ``X[freq]`` (e.g., day of year) and color-codes the lines.
    Useful for comparing within-year (or within-week, etc.) seasonal patterns.

    Args:
      X (pd.DataFrame): DataFrame containing the columns referenced by
        ``period`` and ``freq``.
      y (str): Name of the column in ``X`` to plot on the y-axis.
      period (str): Column name in ``X`` used as the grouping key (e.g., ``"year"``).
      freq (str): Column name in ``X`` used on the x-axis within each period
        (e.g., ``"dayofyear"``).
      ax (matplotlib.axes.Axes | None): Target axes. If ``None``, a new axes is
        created.

    Returns:
      matplotlib.axes.Axes: The axes with the seasonal plot.

    Raises:
      KeyError: If ``period``, ``freq``, or ``y`` are missing from ``X``.
    """
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette(
        "husl",
        n_colors=X[period].nunique(),
    )
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend="linear", ax=None):
    """Plot a (log-frequency) periodogram for a daily series.

    Uses a simple boxcar window and shows spectral density on a log-scaled
    x-axis with labeled seasonal landmarks (weekly, monthly, etc.).

    Args:
      ts (pd.Series): Daily time series (DatetimeIndex or PeriodIndex convertible
        to daily). NaNs are allowed; statsmodels will handle internal details.
      detrend (str): Detrending method passed to ``scipy.signal.periodogram``.
        Common values are ``"linear"`` (default) or ``"constant"``.
      ax (matplotlib.axes.Axes | None): Target axes. If ``None``, a new axes is
        created.

    Returns:
      matplotlib.axes.Axes: The axes with the periodogram.
    """
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling="spectrum",
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    """Scatter/regression plot of ``y_t`` versus ``x_{t-lag}``.

    If ``y`` is ``None``, plots ``x_t`` versus ``x_{t-lag}`` (auto-lag plot).
    Optionally standardizes both axes before plotting and annotates the Pearson
    correlation in the upper-left corner.

    Args:
      x (pd.Series): Series to lag on the x-axis.
      y (pd.Series | None): Optional series for the y-axis. If ``None``,
        ``x`` is used.
      lag (int): Positive integer lag applied to ``x``.
      standardize (bool): If ``True``, standardize both axes to zero mean and
        unit variance before plotting.
      ax (matplotlib.axes.Axes | None): Target axes. If ``None``, a new axes is
        created.
      **kwargs: Additional keyword args forwarded to ``seaborn.regplot`` (e.g.,
        ``scatter_kws``, ``robust``).

    Returns:
      matplotlib.axes.Axes: The axes with the lag plot.

    Raises:
      ValueError: If ``lag`` is not a positive integer.
    """
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(
        color="C3",
    )
    ax = sns.regplot(
        x=x_,
        y=y_,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        lowess=True,
        ax=ax,
        **kwargs,
    )
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    """Create a grid of lag plots for lags 1..``lags``.

    Args:
      x (pd.Series): Series to lag on the x-axis.
      y (pd.Series | None): Optional series for the y-axis. If ``None``,
        auto-lag plots are created.
      lags (int): Maximum lag (inclusive) to plot.
      nrows (int): Number of rows in the grid layout.
      lagplot_kwargs (dict): Keyword arguments forwarded to :func:`lagplot`.
      **kwargs: Additional keyword args forwarded to ``plt.subplots`` (e.g.,
        ``figsize``).

    Returns:
      matplotlib.figure.Figure: The figure containing the grid of lag plots.
    """
    kwargs.setdefault("nrows", nrows)
    kwargs.setdefault("ncols", math.ceil(lags / nrows))
    kwargs.setdefault("figsize", (kwargs["ncols"] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs["nrows"] * kwargs["ncols"])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis("off")
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


def plot_ccf_oil_global_long(
    oil_df: pd.DataFrame,
    glob_resid: pd.Series,
    lags: int = 180,
    agg: str = "W",
    use_returns: bool = True,
    title: Optional[str] = None,
):
    """Plot the CCF between oil prices and global sales residuals.

    Aligns both series daily, optionally converts oil to log-returns, optionally
    aggregates both to a lower frequency (weekly by default), and then plots the
    cross-correlation function for the specified lag window.

    Args:
      oil_df (pd.DataFrame): DataFrame with a ``"dcoilwtico"`` column containing
        oil prices (levels). Must be indexed by date or convertible to daily.
      glob_resid (pd.Series): Global deseasoned sales residuals indexed by date.
      lags (int): Maximum number of (daily) lags to plot on either side. For
        weekly aggregation, ``180`` corresponds to ~26 weeks.
      agg (str | None): Resample rule (e.g., ``"W"``, ``"M"``). If ``None``,
        no aggregation is performed.
      use_returns (bool): If ``True``, use log-returns of oil; otherwise use
        level.
      title (str | None): Optional title to override the default.

    Returns:
      None: The function produces a matplotlib figure and calls ``plt.show()``.

    Notes:
      Positive lags on the x-axis indicate that oil leads residuals.
    """
    # Daily aligned series
    oil_s = _ensure_daily(oil_df["dcoilwtico"].astype(float)).ffill()
    glob_s = _ensure_daily(glob_resid).ffill()

    # Transform: returns or levels
    if use_returns:
        oil_x = np.log(oil_s.replace(0.0, np.nan)).ffill().diff()
        # Glob residuals are already high-frequency; keep as-is
        y_lab = "Global sales residual"
        x_lab = "Oil log-returns"
    else:
        oil_x = oil_s  # continuous level
        y_lab = "Global sales residual"
        x_lab = "Oil level"

    # Optional weekly/monthly aggregation to capture macro timing
    if agg is not None:
        oil_x = oil_x.resample(agg).mean()
        glob_s = glob_s.resample(agg).median()

    df = pd.concat({"glob": glob_s, "oil_x": oil_x}, axis=1).dropna()
    if df.empty:
        print("[warn] No overlap after alignment.")
        return None

    plt.figure(figsize=(9, 3.2))
    plot_ccf(df["oil_x"], df["glob"], lags=lags)
    plt.title(title or f"CCF: {x_lab} (lead) → {y_lab}  |  agg={agg}, lags={lags}")
    plt.xlabel("Lead/lag (positive = oil leads)")
    plt.show()


def summarize_transactions_ccf_allstores(transactions, store_resid, lags=12):
    """Summarize best CCF lag between transactions and residuals per store.

    For each store, computes the correlation between residuals and
    log-differences of transactions across lags ``[-lags, +lags]`` and returns
    the lag with the highest absolute correlation.

    Args:
      transactions (pd.DataFrame): Either:
        * Long: MultiIndex with levels ``("store_nbr", "date")`` and a
          ``"transactions"`` column, or
        * Wide: Index is date and columns are store numbers (or a top level
          ``("transactions", store_nbr)`` which will be sliced).
      store_resid (pd.DataFrame): Wide residual matrix with index as date and
        columns as store numbers.
      lags (int): Maximum absolute lag to evaluate.

    Returns:
      pd.DataFrame: Rows per store with columns
        ``["store_nbr", "best_lag", "corr", "n", "ci95"]``, sorted by
        ``abs(corr)`` descending.
    """
    tx = transactions.copy()
    if isinstance(tx.index, pd.MultiIndex):
        if tx.index.names[0] != "store_nbr":
            tx = tx.reorder_levels(["store_nbr", "date"]).sort_index()

    rows = []
    for s in store_resid.columns:
        res = _ensure_daily(store_resid[s]).ffill()
        try:
            tx_s = tx.xs(s, level="store_nbr")["transactions"]
        except KeyError:
            continue
        tx_s = _ensure_daily(tx_s).ffill()
        tx_ret = np.log1p(tx_s).diff()
        df = pd.concat({"res": res, "tx_ret": tx_ret}, axis=1).dropna()
        if df.empty:
            continue

        best = None
        for k in range(-lags, lags + 1):
            c = df["res"].corr(df["tx_ret"].shift(k))
            if np.isfinite(c):
                if best is None or abs(c) > abs(best[1]):
                    best = (k, c)
        if best:
            n = len(df.dropna())
            ci = 1.96 / np.sqrt(n) if n > 0 else np.nan  # approx 95% threshold
            rows.append(
                {
                    "store_nbr": s,
                    "best_lag": best[0],
                    "corr": best[1],
                    "n": n,
                    "ci95": ci,
                }
            )
    out = pd.DataFrame(rows).sort_values("corr", key=np.abs, ascending=False)
    return out


def plot_ccf_transactions_store(
    transactions, store_resid, store_id, lags=12, min_points=40
):
    """Plot the CCF between transactions and residuals for a given store.

    Args:
      transactions (pd.DataFrame): See :func:`summarize_transactions_ccf_allstores`.
      store_resid (pd.DataFrame): Wide residual matrix with index as date and
        columns as store numbers.
      store_id (int | str): Store identifier (exact or string-matchable).
      lags (int): Maximum lag to plot on either side.
      min_points (int): Minimum overlapping observations required to proceed
        without an early informational warning.

    Returns:
      matplotlib.figure.Figure: The created figure containing the CCF plot.

    Raises:
      KeyError: If ``store_id`` cannot be matched to either ``store_resid`` or
        ``transactions``.
    """
    # Residuals for this store
    store_label = _match_key_in_index_like(store_id, store_resid.columns)
    s_res = _ensure_daily(store_resid[store_label]).ffill()

    # Transactions for this store
    tx = transactions.copy()
    if isinstance(tx.index, pd.MultiIndex):
        if tx.index.names[0] != "store_nbr":
            tx = tx.reorder_levels(["store_nbr", "date"]).sort_index()
        tx_label = _match_key_in_index_like(
            store_id, tx.index.get_level_values("store_nbr").unique()
        )
        tx_s = tx.xs(tx_label, level="store_nbr")["transactions"]
    else:
        tx_s = tx["transactions"]
    tx_s = _ensure_daily(tx_s).ffill()

    # Transforms
    tx_ret = np.log1p(tx_s).diff()
    df = pd.concat({"res": s_res, "tx_ret": tx_ret}, axis=1).dropna()

    # Guardrails and info
    if df.empty or len(df) < max(min_points, lags + 5):
        print(
            f"[warn] Not enough overlap to plot CCF for store {store_id} "
            f"(len={len(df)}, lags={lags}). Try a different store or lower lags."
        )
    else:
        print(
            f"[ok] Overlap: {df.index.min().date()} → {df.index.max().date()}, N={len(df)}"
        )

    # Explicit fig/ax; return fig so Jupyter renders it
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_ccf(df["tx_ret"], df["res"], lags=lags, ax=ax)
    ax.set_title(f"CCF: Transactions (lead) → Residuals (store {store_id})")
    fig.tight_layout()
    return fig


def _demean(s: pd.Series) -> pd.Series:
    """Return a demeaned copy of a series.

    Args:
      s (pd.Series): Input series.

    Returns:
      pd.Series: ``s - s.mean()`` with the same index and name.
    """
    return s - s.mean()


def _apply_ar_filter(series: pd.Series, phis: np.ndarray) -> pd.Series:
    """Apply an AR(p) filter ``1 - phi1 L - ... - phip L^p`` to a (demeaned) series.

    The function subtracts lagged values multiplied by the supplied AR
    coefficients. It assumes the input series has already been demeaned; the
    function does not remove the mean internally.

    Args:
      series (pd.Series): (Demeaned) series to filter.
      phis (np.ndarray): Array of AR coefficients of length ``p``.

    Returns:
      pd.Series: The filtered series aligned to the input index.
    """
    out = series.copy()
    for i, phi in enumerate(phis, start=1):
        out = out - phi * series.shift(i)
    return out


def _fit_ar_phis(x: pd.Series, max_ar: int = 7) -> np.ndarray:
    """Fit AR(p) models and return coefficients for the best AIC model.

    Tries orders ``p=1..max_ar`` using ``statsmodels.tsa.ar_model.AutoReg`` with
    no trend (``trend='n'``) and returns the lag coefficients (``phis``) of the
    model with the lowest AIC. If fitting fails for all orders, returns an
    empty array.

    Args:
      x (pd.Series): Input series (will be ``dropna()``d). Demeaning is not
        performed here; if you need prewhitening, demean before calling.
      max_ar (int): Maximum AR order to consider.

    Returns:
      np.ndarray: Array of AR coefficients of length ``p`` or an empty array if
      no model could be fit successfully.
    """
    x0 = x.dropna()
    best = None
    for p in range(1, max_ar + 1):
        try:
            m = AutoReg(x0, lags=p, trend="n", old_names=False).fit()
            if best is None or m.aic < best[0]:
                # params are lag coeffs only because trend='n'
                phis = m.params.values  # length p
                best = (m.aic, phis)
        except Exception:
            continue
    return np.array([]) if best is None else best[1]


def summarize_onpromo_ccf_by_pair_prewhiten(
    y_resid_wide: pd.DataFrame,
    onpromo_wide: pd.DataFrame,
    max_lead: int = 16,
    max_ar: int = 7,
    min_points: int = 100,
) -> pd.DataFrame:
    """Summarize prewhitened CCF between on-promotion and residuals by series.

    For each ``(store_nbr, family)`` column present in both inputs:
      1) Demean on-promotion and residual series.
      2) Fit AR(p) to on-promotion (``p ≤ max_ar``) by AIC; apply the same AR
         polynomial to both series (prewhitening).
      3) Compute correlations ``corr(res, onp.shift(k))`` for leads
         ``k in [0, max_lead]`` and record the best absolute correlation.

    Args:
      y_resid_wide (pd.DataFrame): Wide residual matrix with columns as a
        MultiIndex ``(store_nbr, family)``.
      onpromo_wide (pd.DataFrame): Wide on-promotion signal with the same
        MultiIndex columns.
      max_lead (int): Maximum positive lead (days) to evaluate for on-promotion.
      max_ar (int): Maximum AR order for prewhitening on-promotion.
      min_points (int): Minimum overlapping observations required per series.

    Returns:
      pd.DataFrame: Rows per ``(store_nbr, family)`` with columns
        ``["store_nbr", "family", "best_lead", "corr", "n", "ci95", "ar_order"]``,
        sorted by ``abs(corr)`` descending.
    """
    common = y_resid_wide.columns.intersection(onpromo_wide.columns)
    rows = []
    for col in common:
        res = _ensure_daily(y_resid_wide[col]).ffill()
        onp = _ensure_daily(onpromo_wide[col]).ffill()

        # Demean both; fit AR on onpromotion only
        onp0 = _demean(onp)
        res0 = _demean(res)
        phis = _fit_ar_phis(onp0, max_ar=max_ar)

        # If we have AR structure, remove it from both series
        x_pw = _apply_ar_filter(onp0, phis) if len(phis) else onp0
        y_pw = _apply_ar_filter(res0, phis) if len(phis) else res0

        df = pd.concat({"res": y_pw, "onp": x_pw}, axis=1).dropna()
        if len(df) < max(min_points, max_lead + 5):
            continue

        best = None
        for k in range(0, max_lead + 1):  # promotions lead
            c = df["res"].corr(df["onp"].shift(k))
            if np.isfinite(c) and (best is None or abs(c) > abs(best[1])):
                best = (k, c)
        if best is None:
            continue

        store_nbr, family = col
        n = len(df)
        rows.append(
            {
                "store_nbr": store_nbr,
                "family": family,
                "best_lead": int(best[0]),
                "corr": float(best[1]),
                "n": int(n),
                "ci95": float(1.96 / np.sqrt(n)) if n > 0 else np.nan,
                "ar_order": int(len(phis)),
            }
        )
    out = pd.DataFrame(rows).sort_values("corr", key=np.abs, ascending=False)
    return out


def plot_ccf_onpromotion_pair_prewhiten(
    onpromo_wide: pd.DataFrame,
    y_resid_wide: pd.DataFrame,
    store_id,
    family,
    lags: int = 16,
    max_ar: int = 7,
):
    """Plot a prewhitened CCF for one (store, family) series.

    Demeans the series, fits AR(p) to on-promotion by AIC (up to ``max_ar``),
    filters both series with the same AR polynomial, and plots the CCF.

    Args:
      onpromo_wide (pd.DataFrame): Wide on-promotion signal with MultiIndex
        columns ``(store_nbr, family)``.
      y_resid_wide (pd.DataFrame): Wide residual matrix with the same column
        structure.
      store_id (int | str): Store identifier (exact or fuzzy match supported).
      family (str): Family identifier (exact or fuzzy match supported).
      lags (int): Maximum lag to display on either side of zero.
      max_ar (int): Maximum AR order for prewhitening on-promotion.

    Returns:
      matplotlib.figure.Figure | None: Figure with the CCF, or ``None`` if
      there are not enough overlapping observations.
    """
    col_key = (
        _match_key_in_index_like(
            store_id, y_resid_wide.columns.get_level_values("store_nbr")
        ),
        _match_key_in_index_like(
            family, y_resid_wide.columns.get_level_values("family")
        ),
    )
    res = _ensure_daily(y_resid_wide[col_key]).ffill()
    onp = _ensure_daily(onpromo_wide[col_key]).ffill()

    onp0 = _demean(onp)
    res0 = _demean(res)
    phis = _fit_ar_phis(onp0, max_ar=max_ar)
    x_pw = _apply_ar_filter(onp0, phis) if len(phis) else onp0
    y_pw = _apply_ar_filter(res0, phis) if len(phis) else res0

    df = pd.concat({"res": y_pw, "onp": x_pw}, axis=1).dropna()
    if len(df) < lags + 5:
        print(
            f"[warn] Not enough overlap for store {store_id}, family {family} (len={len(df)})."
        )
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_ccf(df["onp"], df["res"], lags=lags, ax=ax)
    ax.set_title(
        f"Prewhitened CCF: OnPromotion (lead) → Residuals ({store_id}, {family})"
    )
    fig.tight_layout()
    return fig


def event_study_onpromotion_starts(
    y_resid_wide: pd.DataFrame,
    onpromo_bin_wide: pd.DataFrame,
    store_id,
    family,
    pre_days: int = 7,
    post_days: int = 14,
    min_gap: int = 7,
):
    """Plot average residual response around promotion starts (0→1 transitions).

    Aligns windows around days where the binary on-promotion variable flips from
    0 to 1, enforces a minimum spacing between events, averages the residual
    trajectories, and plots the mean response.

    Args:
      y_resid_wide (pd.DataFrame): Wide residual matrix with MultiIndex columns
        ``(store_nbr, family)``.
      onpromo_bin_wide (pd.DataFrame): Wide **binary** promotion indicator with
        the same columns as ``y_resid_wide``.
      store_id (int | str): Store identifier (exact or fuzzy match supported).
      family (str): Family identifier (exact or fuzzy match supported).
      pre_days (int): Days to include before the event.
      post_days (int): Days to include after the event.
      min_gap (int): Minimum number of days required between consecutive starts
        to avoid heavy overlap of windows.

    Returns:
      matplotlib.figure.Figure | None: The event-study figure, or ``None`` if
      no eligible events/windows are found.
    """
    # Pick the (store, family) column
    s_key = _match_key_in_index_like(
        store_id, y_resid_wide.columns.get_level_values("store_nbr")
    )
    f_key = _match_key_in_index_like(
        family, y_resid_wide.columns.get_level_values("family")
    )
    col = (s_key, f_key)

    # Daily, forward-filled residuals and binary promotion
    res = _ensure_daily(y_resid_wide[col]).ffill()
    onp_b = _ensure_daily(onpromo_bin_wide[col]).ffill()

    # Find starts (0 -> 1)
    starts_mask = (onp_b.astype("int8").diff() == 1).fillna(False)
    start_dates = starts_mask.index[starts_mask]

    if len(start_dates) == 0:
        print(f"[info] No starts for ({store_id}, {family}).")
        return None

    # Enforce minimum spacing between starts without using Timestamp.min
    valid_starts = []
    last_dt = None
    gap = pd.Timedelta(min_gap, "D")
    for dt in start_dates:
        if (last_dt is None) or ((dt - last_dt) >= gap):
            valid_starts.append(dt)
            last_dt = dt

    if not valid_starts:
        print(f"[info] No starts after min_gap for ({store_id}, {family}).")
        return None

    # Build matrix of residuals around each start
    mat = []
    for dt in valid_starts:
        # Target index for this window
        idx = pd.date_range(
            dt - pd.Timedelta(pre_days, "D"),
            dt + pd.Timedelta(post_days, "D"),
            freq="D",
        )
        seg = res.reindex(idx)
        if seg.isna().any():
            # Skip partial windows
            continue
        mat.append(seg.values)

    if not mat:
        print(f"[info] Not enough full windows for ({store_id}, {family}).")
        return None

    M = np.vstack(mat)
    avg = M.mean(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(-pre_days, post_days + 1)
    ax.plot(x, avg, marker="o")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(
        f"Event study: residual response to promo START ({store_id}, {family})"
    )
    ax.set_xlabel("Days from start (0 = first promo day)")
    ax.set_ylabel("Residual (avg across events)")
    fig.tight_layout()
    return fig
