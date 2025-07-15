"""
Feature engineering, preprocessing helpers and sklearn-compatible
transformers for the Titanic competition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Tuple, Any


# ----------------------------------------------------------------------
#  Core preprocessing functions
# ----------------------------------------------------------------------
def fit_preprocessor(df, min_freq: float = 0.01):
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
        lambda r: medians_age_train.loc[(r.Pclass, r.Sex, r.Title)] if pd.isna(r.Age) else r.Age,
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
        index=df.index
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
    df,
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
):
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
        allowed = common_levels[col] # set of “frequent” values from training
        df[col] = df[col].apply(lambda val: val if val in allowed else "Other")
    df_cat = ohe.transform(df[cat_cols])
    
    # — 15) Assemble final X
    X = pd.DataFrame(
        np.hstack([df[num_cols], df_cat]),
        columns=list(num_cols) + list(ohe.get_feature_names_out(cat_cols)),
        index=df.index
    )
    
    # — 16) Build a YDF‐compatible copy
    df_ydf = df.copy()
    if y is not None:
        df_ydf["Survived"] = y.values
    
    return X, y, df_ydf

# ----------------------------------------------------------------------
#  Convenience helper
# ----------------------------------------------------------------------
def preprocess_fold(df_train_raw: pd.DataFrame,
                    df_valid_raw: pd.DataFrame):
    """
    Fits the full pipeline only on df_train_raw and uses the learned
    params to transform both train & valid parts, returning:
        X_tr, y_tr  – numpy/pandas for XGBoost
        X_va, y_va
        df_tr_ydf, df_va_ydf – DataFrames for YDF-GBT
    """
    # ---- fit on the training portion ----
    X_tr, y_tr, df_tr_ydf, params = fit_preprocessor(df_train_raw)

    # ---- transform the validation portion ----
    X_va, y_va, df_va_ydf = transform_preprocessor(df_valid_raw, **params)
    
    return X_tr, y_tr, df_tr_ydf, X_va, y_va, df_va_ydf

# ----------------------------------------------------------------------
#  sklearn Transformers for Pipelines
# ----------------------------------------------------------------------
class PreprocXGB(BaseEstimator, TransformerMixin):
    """Returns the fully one-hot-encoded numeric matrix."""
    def fit(self, X, y=None):
        self.X_template_,  _, _, self.params_ = fit_preprocessor(X)
        return self
    def transform(self, X):
        X_num, _, _ = transform_preprocessor(X, **self.params_)
        return X_num            # numeric / one-hot view

class PreprocYDF(BaseEstimator, TransformerMixin):
    """Returns df_ydf view -> numeric + raw string categoricals."""
    def fit(self, X, y=None):
        _, _, self.df_tmpl_, self.params_ = fit_preprocessor(X)
        return self
    def transform(self, X):
        _, _, df_ydf = transform_preprocessor(X, **self.params_)
        return df_ydf.drop(columns=["Survived"])   # YDFWrapper adds label later