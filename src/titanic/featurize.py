"""Feature engineering and preprocessing utilities for the Titanic project.

This module contains two layers of functionality:

1. **Stateless helpers** that *fit* (`fit_preprocessor`) or *apply*
   (`transform_preprocessor`) the full feature-engineering pipeline.
2. **sklearn-compatible transformers** (`PreprocXGB`, `PreprocYDF`) so the same
   pipeline can be embedded inside GridSearch-, cross-validation-, and stacking
   workflows.

The pipeline reproduces all engineering steps documented in the notebook,
including:

* text extraction (titles, surnames),
* family / cabin / ticket derived signals,
* missing-value imputation,
* age bucketing,
* log-fare scaling,
* rare-category pooling and one-hot encoding.

Everything is pure pandas/numpy so it can run on CPU-only environments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------
#  Core preprocessing functions
# ----------------------------------------------------------------------
def fit_preprocessor(
    df: pd.DataFrame,
    *,
    min_freq: float = 0.01,
) -> Tuple[pd.DataFrame, pd.Series | None, pd.DataFrame, Dict[str, Any]]:
    """Fit the full Titanic feature pipeline on *df*.

    Args:
        df: Raw Titanic DataFrame.  Must include **all** original columns
            (`PassengerId`, `Name`, `Ticket`, `Cabin`, etc.).  May or may not
            include the `"Survived"` target.
        min_freq: Minimum fraction (0–1) a categorical level must appear
            in *df* in order **not** to be collapsed to the `"Other"` bucket.

    Returns:
        X: One-hot / numeric feature matrix suitable for scikit-learn and
           XGBoost.  Index is preserved from *df*.
        y: Target vector if present in *df*, otherwise ``None``. df_ydf: Pandas
        DataFrame view (numeric plus *raw string*
           categoricals) required by YDF learners.
        params: Dictionary of fitted objects and training statistics
           (scaler, encoder, medians, etc.).  Pass verbatim to
           :func:`transform_preprocessor`.
    """

    df = df.copy()

    # — 1) Extract titles & surnames
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["Surname"] = df["Name"].str.split(",", expand=True)[0]

    # — 2) Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # — 3) Cabin features
    df["CabinCount"] = df["Cabin"].fillna("").apply(lambda s: len(s.split()))
    df["Deck"] = df["Cabin"].str[0].fillna("Missing")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["CabinNumber"] = (
        df["Cabin"].str.extract(r"(\d+)", expand=False).fillna(0).astype(int)
    )

    # — 4) Ticket features
    df["TicketNumber"] = (
        df["Ticket"].str.extract(r"(\d+)$", expand=False).fillna(0).astype(int)
    )
    df["TicketItem"] = (
        df["Ticket"]
        .str.split()
        .apply(lambda parts: "_".join(parts[:-1]) if len(parts) > 1 else "NONE")
    )

    # — 5) Impute Embarked (most frequent)
    mode_embarked_train = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(mode_embarked_train)

    # — 6) Impute Fare (median)
    median_fare_train = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(median_fare_train)

    # — 7) Impute Age (median by Pclass×Sex×Title)
    medians_age_train = df.groupby(["Pclass", "Sex", "Title"])["Age"].median()
    global_median_age_train = df["Age"].median()
    df["Age"] = df.apply(
        lambda r: medians_age_train.loc[(r.Pclass, r.Sex, r.Title)]
        if pd.isna(r.Age)
        else r.Age,
        axis=1,
    )

    # — 8) Bucket Age into decades
    bins = list(range(0, 81, 10))
    labels = [f"{i}-{i + 10}" for i in bins[:-1]]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    df["AgeGroup"] = (
        df["AgeGroup"].cat.add_categories("Missing").fillna("Missing").astype(str)
    )

    # — 9) Log-transform Fare & scale
    df["LogFare"] = np.log1p(df["Fare"])
    scaler = StandardScaler().fit(df[["LogFare"]])
    df["FareScaled"] = scaler.transform(df[["LogFare"]])

    # — 10) Frequency-encode surname
    surname_counts = df["Surname"].value_counts()
    df["SurnameCount"] = df["Surname"].map(surname_counts).fillna(0)

    # — 11) Drop raw columns
    df.drop(
        ["PassengerId", "Name", "Surname", "Cabin", "Ticket", "Age", "Fare", "LogFare"],
        axis=1,
        inplace=True,
    )
    # — 12) Separate target
    if "Survived" in df.columns:
        y = df.pop("Survived")
    else:
        y = None

    # — 13) Split numeric vs categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # — 14) OneHotEncode infrequent
    common_levels = {}
    for col in cat_cols:
        # Compute absolute threshold:
        n_rows = len(df)
        thresh = int(min_freq * n_rows)
        # Count frequencies in the training set
        counts = df[col].value_counts()
        # Keep only levels where count >= thresh
        frequent = set(counts[counts >= thresh].index.tolist())
        common_levels[col] = frequent

        # Replace anything outside that set with "Other"
        df[col] = df[col].apply(lambda val: val if val in frequent else "Other")

    ohe = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    ).fit(df[cat_cols])
    df_cat = ohe.transform(df[cat_cols])

    # — 15) Assemble final X
    X = pd.DataFrame(
        np.hstack([df[num_cols], df_cat]),
        columns=list(num_cols) + list(ohe.get_feature_names_out(cat_cols)),
        index=df.index,
    )

    # — 16) Build a YDF‐compatible copy
    df_ydf = df.copy()
    if y is not None:
        df_ydf["Survived"] = y.values

    params = {
        "scaler": scaler,
        "ohe": ohe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "medians_age_train": medians_age_train,
        "global_median_age_train": global_median_age_train,
        "mode_embarked_train": mode_embarked_train,
        "median_fare_train": median_fare_train,
        "common_levels": common_levels,
        "surname_counts": surname_counts,
    }

    return X, y, df_ydf, params


def transform_preprocessor(
    df: pd.DataFrame,
    *,
    scaler,
    ohe,
    num_cols,
    cat_cols,
    medians_age_train,
    global_median_age_train,
    mode_embarked_train,
    median_fare_train,
    common_levels,
    surname_counts,
) -> Tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Apply a *fitted* pipeline to new data.

    Parameters
    ----------
    df
        Raw Titanic frame (same schema as training).
    scaler, ohe, num_cols, cat_cols, medians_age_train, global_median_age_train,
    mode_embarked_train, median_fare_train, common_levels, surname_counts
        Objects / statistics produced by :func:`fit_preprocessor`.

    Returns:
        X: One-hot / numeric features.
        y: Target (or ``None`` if missing).
        df_ydf: YDF-compatible DataFrame view.
    """

    df = df.copy()

    # — 1) Extract titles & surnames
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["Surname"] = df["Name"].str.split(",", expand=True)[0]

    # — 2) Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # — 3) Cabin features
    df["CabinCount"] = df["Cabin"].fillna("").apply(lambda s: len(s.split()))
    df["Deck"] = df["Cabin"].str[0].fillna("Missing")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["CabinNumber"] = (
        df["Cabin"].str.extract(r"(\d+)", expand=False).fillna(0).astype(int)
    )

    # — 4) Ticket features
    df["TicketNumber"] = (
        df["Ticket"].str.extract(r"(\d+)$", expand=False).fillna(0).astype(int)
    )
    df["TicketItem"] = (
        df["Ticket"]
        .str.split()
        .apply(lambda parts: "_".join(parts[:-1]) if len(parts) > 1 else "NONE")
    )

    # — 5) Impute Embarked (using train set most frequent)
    df["Embarked"] = df["Embarked"].fillna(mode_embarked_train)

    # — 6) Impute Fare (median)
    df["Fare"] = df["Fare"].fillna(median_fare_train)

    # — 7) Impute Age (median by Pclass×Sex×Title from train)
    df["Age"] = df.apply(
        lambda r: (
            medians_age_train.get((r.Pclass, r.Sex, r.Title), global_median_age_train)
            if pd.isna(r.Age)
            else r.Age
        ),
        axis=1,
    )

    # — 8) Bucketize Age into decades
    bins = list(range(0, 81, 10))
    labels = [f"{i}-{i + 10}" for i in bins[:-1]]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    df["AgeGroup"] = (
        df["AgeGroup"].cat.add_categories("Missing").fillna("Missing").astype(str)
    )

    # — 9) Log-transform Fare & scale
    df["LogFare"] = np.log1p(df["Fare"])
    df["FareScaled"] = scaler.transform(df[["LogFare"]])

    # — 10) Frequency-encode surname
    df["SurnameCount"] = df["Surname"].map(surname_counts).fillna(0)

    # — 11) Drop raw columns
    df.drop(
        ["PassengerId", "Name", "Surname", "Cabin", "Ticket", "Age", "Fare", "LogFare"],
        axis=1,
        inplace=True,
    )

    # — 12) Separate target
    if "Survived" in df.columns:
        y = df.pop("Survived")
    else:
        y = None

    # — 13) Split numeric vs categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # — 14) OneHotEncode
    for col in cat_cols:
        allowed = common_levels[col]  # set of “frequent” values from training
        df[col] = df[col].apply(lambda val: val if val in allowed else "Other")
    df_cat = ohe.transform(df[cat_cols])

    # — 15) Assemble final X
    X = pd.DataFrame(
        np.hstack([df[num_cols], df_cat]),
        columns=list(num_cols) + list(ohe.get_feature_names_out(cat_cols)),
        index=df.index,
    )

    # — 16) Build a YDF‐compatible copy
    df_ydf = df.copy()
    if y is not None:
        df_ydf["Survived"] = y.values

    return X, y, df_ydf


# ----------------------------------------------------------------------
#  Convenience helper
# ----------------------------------------------------------------------
def preprocess_fold(
    df_train_raw: pd.DataFrame,
    df_valid_raw: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """Leak-free preprocessing for a single CV fold.

    The function *fits* the pipeline **only** on *df_train_raw* and then
    applies the learned parameters to both train and validation splits.

    Args:
        df_train_raw: Training portion of the raw Titanic data (must
            include `"Survived"`).
        df_valid_raw: Validation portion of the raw Titanic data (must
            include `"Survived"`).

    Returns:
        X_tr, y_tr: One-hot / numeric matrix and target for the training
            split.
        df_tr_ydf: YDF-compatible view of the training split.
        X_va, y_va: One-hot / numeric matrix and target for the validation
            split.
        df_va_ydf: YDF-compatible view of the validation split.
    """

    # Fit on the training portion
    X_tr, y_tr, df_tr_ydf, params = fit_preprocessor(df_train_raw)

    # Transform the validation portion
    X_va, y_va, df_va_ydf = transform_preprocessor(df_valid_raw, **params)

    return X_tr, y_tr, df_tr_ydf, X_va, y_va, df_va_ydf


# ----------------------------------------------------------------------
#  sklearn Transformers for Pipelines
# ----------------------------------------------------------------------
class PreprocXGB(BaseEstimator, TransformerMixin):
    """sklearn transformer that outputs the *numeric/one-hot* matrix.

    Designed for models (e.g. XGBoost, logistic regression) that consume
    a dense numeric feature array.

    The transformer *internally* caches all fitted preprocessing objects
    so that consecutive folds in a cross-validation loop reuse the same
    expensive work.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N802
        """Fit the preprocessing pipeline on *X*."""
        self._X_template, _, _, self.params_ = fit_preprocessor(X)
        return self

    def transform(self, X: pd.DataFrame):  # noqa: N802
        """Transform *X* into the numeric/one-hot matrix."""
        X_num, _, _ = transform_preprocessor(X, **self.params_)
        return X_num


class PreprocYDF(BaseEstimator, TransformerMixin):
    """sklearn transformer that outputs the *DataFrame* view for YDF.

    Returns the numeric columns plus **raw string categoricals** (exact
    format expected by `ydf.GradientBoostedTreesLearner`).  The `"Survived"`
    column is intentionally dropped; YDFWrapper will add it during
    `.fit()`.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N802
        """Fit the preprocessing pipeline on *X*."""
        _, _, self._df_tmpl, self.params_ = fit_preprocessor(X)
        return self

    def transform(self, X: pd.DataFrame):  # noqa: N802
        """Transform *X* into a YDF-ready DataFrame (without the label)."""
        _, _, df_ydf = transform_preprocessor(X, **self.params_)
        return df_ydf.drop(columns=["Survived"], errors="ignore")
