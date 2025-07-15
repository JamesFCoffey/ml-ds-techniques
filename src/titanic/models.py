"""
Model wrappers and utilities.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class YDFWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps a ydf.GradientBoostedTreesLearner so it can be used with sklearn's
    cross_val_predict, cross_val_score, etc.

    Parameters
    ----------
    learner : ydf.GradientBoostedTreesLearner
        A *configured but un-trained* learner instance.
    """

    def __init__(self, learner: Any):
        self.learner = learner

    def fit(self, X: pd.DataFrame, y):
        if not hasattr(X, "columns"):
            raise ValueError("YDFWrapper.fit expects a pandas DataFrame.")

        self.classes_ = np.unique(y)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        df_train = X.copy()
        df_train["Survived"] = y.values if hasattr(y, "values") else y
        self.model_ = self.learner.train(df_train, verbose=0)
        return self

    def predict(self, X):
        proba = self._predict_proba_raw(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        p1 = self._predict_proba_raw(X)
        return np.vstack([1 - p1, p1]).T

    def _predict_proba_raw(self, X) -> np.ndarray:
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return self.model_.predict(X)