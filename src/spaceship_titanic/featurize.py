"""Feature engineering utilities for the Spaceship Titanic competition.

This module turns the raw Kaggle tables into model-ready numeric matrices.
It exposes:

* Regular-expression constants that parse cabin strings and passenger IDs.
* :data:`SPEND`, the list of on-board spending columns used repeatedly
  throughout the feature pipeline.
* :func:`_row_featurize`, a stateless helper that creates per-row features
  without any look-ahead.
* :class:`SpaceshipTransformer`, a scikit-learn‐compatible transformer that
  handles imputations, binning, rare-category collapsing, one-hot encoding,
  scaling and optional column dropping.

Typical usage::

    tfm = SpaceshipTransformer(min_freq=0.01).fit(train_df)
    X_train = tfm.transform(train_df)
    X_test  = tfm.transform(test_df)

The transformer is **leak-free** by construction—no information from the test
rows is used when determining bin edges, rare categories, or group-level
aggregations.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Parses the *Cabin* string into deck, numeric cabin and side.
CABIN_RE = re.compile(r"(?P<deck>[A-Z])/(?P<num>\d+)/(?P<side>[PS])")

# Captures the four-digit family/group number from PassengerId.
GROUP_RE = re.compile(r"(\d{4})_(\d{2})")

# Column names that represent monetary spend on board.
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _row_featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-row features **without** using information from other rows.

    Args:
        df: Raw dataframe straight from ``train.csv`` or ``test.csv``.

    Returns:
        pd.DataFrame: A new frame with
          * deck/side/cabin-number extracted,
          * group size, total spend and spend ratios,
          * binary missing-value flags, engineered crosses, and
          * identifier columns dropped.
    """

    df = df.copy()

    # Cabin split
    cabin = df["Cabin"].str.extract(CABIN_RE)
    df["Deck"] = cabin["deck"]  # NaN kept as NaN
    df["Side"] = cabin["side"]
    df["CabinNum"] = cabin["num"].astype(float)

    # Group id / size
    gnum = df["PassengerId"].str.extract(GROUP_RE).astype(int)[0]
    df["GroupNum"] = gnum
    df["GroupSize"] = df.groupby("GroupNum")["PassengerId"].transform("size")

    # Spending raw / ratios
    df[SPEND] = df[SPEND].fillna(0)
    df["TotalSpend"] = df[SPEND].sum(axis=1)
    df["SpendPerPerson"] = df["TotalSpend"] / df["GroupSize"].clip(lower=1)

    for col in SPEND:
        df[f"{col}Ratio"] = np.where(
            df["TotalSpend"] > 0, df[col] / df["TotalSpend"], 0.0
        )

    # Binary flags (keep NaN for flag first)
    df["CryoSleep_NA"] = df["CryoSleep"].isna().astype(int)
    df["VIP_NA"] = df["VIP"].isna().astype(int)
    df["CabinNum_NA"] = df["CabinNum"].isna().astype(int)

    df["CryoSleep"] = df["CryoSleep"].fillna(0).astype(int)
    df["VIP"] = df["VIP"].fillna(0).astype(int)
    df["CabinNum"] = df["CabinNum"].fillna(-1)

    # Surname
    df["Surname"] = df["Name"].str.split().str[-1]

    # Categorical NaNs
    for c in ["HomePlanet", "Destination", "Deck", "Side", "Surname"]:
        df[c] = df[c].fillna("Missing").astype(str)

    # Crosses
    df["DeckSide"] = df["Deck"] + "_" + df["Side"]
    df["HomeDest"] = df["HomePlanet"] + "_" + df["Destination"]

    # Drop IDs
    df = df.drop(columns=["Cabin", "PassengerId", "Name"])

    return df


class SpaceshipTransformer(BaseEstimator, TransformerMixin):
    """End-to-end tabular pre-processor for Spaceship Titanic.

    The transformer:

    * Calls :func:`_row_featurize` to generate first-order features.
    * Builds group-level statistics and numeric bin edges during :py:meth:`fit`.
    * Applies those statistics, one-hot encodes categoricals, scales total
      spend, and optionally drops user-supplied columns during
      :py:meth:`transform`.
    """

    def __init__(
        self,
        min_freq=0.01,
        n_age_bins=6,
        n_spend_bins=3,
        n_cabin_bins=5,
        n_tot_bins=5,
        n_group_bins=5,
        use_fixed_group_bins=True,
        drop_cols=None,
    ):
        """Instantiate the transformer with hyper-parameters.

        Args:
            min_freq: Minimum category frequency (fraction of rows) before a
              label is lumped into ``"Other"``.
            n_age_bins: Quantile bins for :data:`Age`.
            n_spend_bins: Quantile bins for each spend column.
            n_cabin_bins: Quantile bins for :data:`CabinNum`.
            n_tot_bins: Quantile bins for :data:`TotalSpend`.
            n_group_bins: Bins for :data:`GroupSize`.
            use_fixed_group_bins: If ``True`` use domain‐knowledge cut-points
              ``[1, 2, 4, 10]`` rather than quantiles.
            drop_cols: List of column names (or OHE dummy names) to remove
              after feature construction.  Defaults to ``["GroupNum"]``.
        """

        self.min_freq = min_freq
        self.n_age_bins = n_age_bins
        self.n_spend_bins = n_spend_bins
        self.n_cabin_bins = n_cabin_bins
        self.n_tot_bins = n_tot_bins
        self.n_group_bins = n_group_bins
        self.use_fixed_group_bins = use_fixed_group_bins
        self.fixed_grp_edges_ = [0, 1, 2, 4, 10, np.inf]  # domain-knowledge edges
        self.drop_cols = drop_cols or ["GroupNum"]  # list of col names

    def fit(self, X, y=None):
        """Learn bin edges, group aggregations and encoders from *X*.

        Args:
            X: Raw feature dataframe.
            y: Ignored (present only for scikit-learn compatibility).

        Returns:
            SpaceshipTransformer: ``self`` to allow method chaining.
        """

        df = _row_featurize(X)

        # surname counts
        self.surname_cnt_ = df["Surname"].value_counts().to_dict()

        # numeric bins / imputers
        self.age_median_ = df["Age"].median()
        self.age_edges_ = np.quantile(
            df["Age"].fillna(self.age_median_), np.linspace(0, 1, self.n_age_bins + 1)
        )

        nonzero_tot = df.loc[df["TotalSpend"] > 0, "TotalSpend"]
        self.tot_edges_ = np.quantile(
            nonzero_tot, np.linspace(0, 1, self.n_spend_bins + 1)
        )

        nonzero_pp = df.loc[df["SpendPerPerson"] > 0, "SpendPerPerson"]
        self.spp_edges_ = np.quantile(
            nonzero_pp, np.linspace(0, 1, self.n_spend_bins + 1)
        )

        if self.use_fixed_group_bins:
            self.group_edges_ = self.fixed_grp_edges_
        else:
            self.group_edges_ = np.quantile(
                df["GroupSize"], np.linspace(0, 1, self.n_group_bins + 1)
            )

        self.spend_edges_ = {}
        for col in SPEND:
            nz = df.loc[df[col] > 0, col]
            q = np.linspace(0, 1, self.n_spend_bins + 1)
            self.spend_edges_[col] = np.quantile(nz if len(nz) else [0], q)

        mask = df["CabinNum"] >= 0
        self.cabinnum_edges_ = np.quantile(
            df.loc[mask, "CabinNum"], np.linspace(0, 1, self.n_cabin_bins + 1)
        )

        # group agg
        grp = df.groupby("GroupNum")
        self.group_age_median_ = grp["Age"].median().to_dict()
        self.group_cryo_frac_ = grp["CryoSleep"].mean().to_dict()
        self.family_age_gap_ = grp["Age"].agg(lambda s: s.max() - s.min()).to_dict()

        # rare-cat tables
        self.freq_ = {c: df[c].value_counts() for c in df.select_dtypes("object")}

        # one preview for OHE / scaler
        tmp = self._apply_transforms(df)
        self.cat_cols_ = tmp.select_dtypes("object").columns.tolist()
        self.num_cols_ = tmp.select_dtypes(["int64", "float64"]).columns.tolist()

        self.ohe_ = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        ).fit(tmp[self.cat_cols_])
        self.scaler_ = StandardScaler().fit(tmp[["TotalSpend"]])

        full_names = np.concatenate(
            [self.num_cols_, self.ohe_.get_feature_names_out(self.cat_cols_)]
        )
        self._post_drop_idx_ = [
            i for i, f in enumerate(full_names) if f in self.drop_cols
        ]

        return self

    def transform(self, X):
        """Apply the learned transformations to *X*.

        Args:
            X: Raw feature dataframe.

        Returns:
            np.ndarray: Dense 2-D numeric array ready for model consumption.
        """

        df = self._apply_transforms(_row_featurize(X))

        X_cat = self.ohe_.transform(df[self.cat_cols_])
        df["TotalSpend"] = self.scaler_.transform(df[["TotalSpend"]])
        X_num = df[self.num_cols_].to_numpy()

        X = np.hstack([X_num, X_cat])

        if self._post_drop_idx_:
            X = np.delete(X, self._post_drop_idx_, axis=1)

        return X

    def _apply_transforms(self, d):
        """Apply all learned transformations to a feature-engineered frame.

        This helper is called by :meth:`transform` after the raw dataframe has
        been converted to first-order features via :func:`_row_featurize`.  It
        performs imputations, numeric binning, domain buckets, interaction
        terms, group-level look-ups, rare-category collapsing, and final
        column dropping.

        Args:
            d: DataFrame produced by :func:`_row_featurize`. The frame may
                contain NaNs, unseen category labels, or group IDs that did
                not appear in the training split.

        Returns:
            pd.DataFrame: A fully processed dataframe where
              * all numeric features are imputed and, when applicable, binned
                into quantiles or domain buckets,
              * categorical columns have had rare labels consolidated into
                ``"Other"``,
              * interaction columns (e.g. ``Cryo_x_Age``) and group-level
                statistics (e.g. ``GroupAgeMedian``) are added,
              * and any columns listed in ``self.drop_cols`` are removed.
              The result is free of NaNs and ready for one-hot encoding /
              scaling.

        Notes:
            * The method relies on attributes set during :meth:`fit`
              (e.g. ``self.age_edges_`` or ``self.spend_edges_``); therefore
              calling it before fitting will raise ``AttributeError``.
            * Cryo-sleep interactions are implemented as simple pairwise
              products rather than full polynomial features for efficiency.
            * Missing groups that appear only in validation or test data are
              filled with sentinel value ``-1`` after lookup.
        """

        d = d.copy()

        # Age impute + bin
        d["Age"] = d["Age"].fillna(self.age_median_)
        d["Age_qbin"] = pd.cut(
            d["Age"], self.age_edges_, labels=False, include_lowest=True
        )
        # Age domain buckets (children / teen / YA / 26-30 / 31-50 / 51+)
        d["Age_group_dom"] = pd.cut(
            d["Age"],
            bins=[0, 12, 18, 25, 30, 50, 100],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype(int)

        # family size via surname (if NA, assume alone)
        d["FamilySize"] = d["Surname"].map(self.surname_cnt_).fillna(1)

        # luxury spending
        d["LuxurySpend"] = d["Spa"] + d["VRDeck"]

        # deck geometry
        deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
        d["DeckLevel"] = d["Deck"].map(deck_map).fillna(0).astype(int)
        d["CabinSection"] = (
            pd.cut(d["CabinNum"], bins=[-1, 300, 600, 900, 2000], labels=[0, 1, 2, 3])
            .cat.add_categories([4])
            .fillna(4)
            .astype(int)
        )
        d["DeckDistMid"] = (d["DeckLevel"] - 4).abs()
        d["PremiumCabin"] = ((d["DeckLevel"] <= 3) & (d["CabinSection"] == 1)).astype(
            int
        )

        # solo flag
        d["IsSolo"] = (d["GroupSize"] == 1).astype(int)

        # GroupSize bin
        d["GroupSize_bin"] = pd.cut(
            d["GroupSize"], self.group_edges_, labels=False, include_lowest=True
        )

        for col, edges in [
            ("TotalSpend", self.tot_edges_),
            ("SpendPerPerson", self.spp_edges_),
        ]:
            bin_col = f"{col}_bin"
            d[bin_col] = 0  # 0  →  “zero spend”
            mask = d[col] > 0
            d.loc[mask, bin_col] = (
                pd.cut(d.loc[mask, col], bins=edges, labels=False, include_lowest=True)
                + 1  # shift so 1/2/3 = low/med/high
            )

        # Spend bins
        for col in SPEND:
            d[f"{col}_bin"] = 0
            mask = d[col] > 0
            d.loc[mask, f"{col}_bin"] = (
                pd.cut(
                    d.loc[mask, col],
                    bins=self.spend_edges_[col],
                    labels=False,
                    include_lowest=True,
                )
                + 1  # 1=low,2=med,3=high
            )

        # CabinNum bins
        mask = d["CabinNum"] >= 0
        d["CabinNum_qbin"] = -1
        d.loc[mask, "CabinNum_qbin"] = (
            pd.cut(
                d.loc[mask, "CabinNum"],
                self.cabinnum_edges_,
                labels=False,
                include_lowest=True,
            )
            .fillna(-1)
            .astype(int)
        )

        # CryoSleep interactions (cheap pairwise)
        for feat in ["Age", "DeckLevel", "GroupSize"]:
            d[f"Cryo_x_{feat}"] = d["CryoSleep"] * d[feat]

        # Group-level
        d["GroupAgeMedian"] = d["GroupNum"].map(self.group_age_median_)
        d["GroupCryoFrac"] = d["GroupNum"].map(self.group_cryo_frac_)
        d["FamilyAgeGap"] = d["GroupNum"].map(self.family_age_gap_)

        # Missing groups (i.e. groups that appear only in the validation / test set) will get NaN
        d[["GroupAgeMedian", "GroupCryoFrac", "FamilyAgeGap"]] = d[
            ["GroupAgeMedian", "GroupCryoFrac", "FamilyAgeGap"]
        ].fillna(-1)

        # Rare-category collapse
        for col, freq in self.freq_.items():
            rare = freq[freq < self.min_freq * len(freq)].index
            d[col] = np.where(d[col].isin(freq.index), d[col], "Other")
            d[col] = np.where(d[col].isin(rare), "Other", d[col])

        d = d.drop(columns=[c for c in self.drop_cols if c in d.columns])

        d = d.fillna(-1)  # guarantees no NaN leaves the transformer

        return d

    def get_feature_names_out(self, input_features=None):
        """Return the feature names produced by :py:meth:`transform`.

        scikit-learn calls this to align SHAP values, etc.

        Args:
            input_features: Unused; kept for API fidelity.

        Raises:
            RuntimeError: If the transformer is called before :py:meth:`fit`.

        Returns:
            np.ndarray: Array of string feature names (after one-hot encoding
            and any post-drop column removal).
        """

        if not hasattr(self, "ohe_"):
            raise RuntimeError("Must call .fit() first")
        names = np.concatenate(
            [self.num_cols_, self.ohe_.get_feature_names_out(self.cat_cols_)]
        )
        if hasattr(self, "_post_drop_idx_"):
            names = np.delete(names, self._post_drop_idx_)
        return names
