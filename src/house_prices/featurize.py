# src/house_prices/featurize.py

"""
Module for data preprocessing and feature engineering pipeline for the House
Prices competition.

This module provides a function to split the raw DataFrame into train/validation
sets, handle missing values, encode categorical variables, scale numeric
features, and prepare inputs for both scikit-learn models and TensorFlow
Decision Forests models (YDF).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    ColumnTransformer,
]:
    """
    Split data and build preprocessing pipelines.

    This function:
      1. Splits the DataFrame into training and validation sets.
      2. Imputes missing values (median for numeric, constant 'MISSING' for
         categorical).
      3. Optionally scales numeric features.
      4. One-hot encodes categorical features.
      5. Log-transforms the target variable.
      6. Returns both arrays and DataFrames ready for modeling: - X_train_prep,
         X_valid_prep: numpy arrays for scikit-learn models or raw input. -
         y_train, y_valid: log-transformed target series. - df_train_ydf,
         df_valid_ydf: pandas DataFrames with feature columns and 'SalePrice'
         for YDF learners. - preprocessor: the fitted ColumnTransformer for
         later inference on test data.

    Args:
        df (pd.DataFrame):
            DataFrame containing features and a 'SalePrice' column as the
            target.
        test_size (float):
            Proportion of data to hold out for validation (default: 0.2).
        random_state (int):
            Random seed for reproducibility of the train/validation split
            (default: 42).
        scale_numeric (bool):
            Whether to apply StandardScaler to numeric features (default:
            False).

    Returns:
        X_train_prep (np.ndarray):
            Preprocessed training feature array.
        X_valid_prep (np.ndarray):
            Preprocessed validation feature array.
        y_train (pd.Series):
            Log1p-transformed target series for training.
        y_valid (pd.Series):
            Log1p-transformed target series for validation.
        df_train_ydf (pd.DataFrame):
            Training DataFrame with preprocessed features and 'SalePrice' for
            YDF.
        df_valid_ydf (pd.DataFrame):
            Validation DataFrame with preprocessed features and 'SalePrice' for
            YDF.
        preprocessor (ColumnTransformer):
            The fitted transformer for numeric and categorical columns, used for
            test-time transformations.
    """
    # 1) Split
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3) Numeric pipeline: median impute + optional scale
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(num_steps)

    # 4) Categorical pipeline: fill MISSING + one-hot
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 5) Combine
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # 6) Fit & transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_valid_prep = preprocessor.transform(X_valid)

    # 7) Grab feature names to rebuild DataFrames for YDF
    feat_names = []
    # numeric names stay the same
    feat_names += num_cols
    # categorical names from one-hot
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_cols)
    )
    feat_names += cat_names.tolist()

    # 8) Scale y
    y_train = np.log1p(y_train)
    y_valid = np.log1p(y_valid)

    # 9) Build DataFrames for YDF learners
    df_train_ydf = pd.DataFrame(X_train_prep, columns=feat_names)
    df_train_ydf["SalePrice"] = y_train.reset_index(drop=True)
    df_valid_ydf = pd.DataFrame(X_valid_prep, columns=feat_names)
    df_valid_ydf["SalePrice"] = y_valid.reset_index(drop=True)

    return (
        X_train_prep,
        X_valid_prep,
        y_train,
        y_valid,
        df_train_ydf,
        df_valid_ydf,
        preprocessor,
    )
