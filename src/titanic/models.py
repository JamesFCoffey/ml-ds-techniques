"""Lightweight sklearn wrappers around YDF learners.

`ydf` (Yet-another Decision Forests) models do **not** implement the
scikit-learn estimator API.  This module provides a thin adapter—
:class:`YDFWrapper`—so they can be dropped into any sklearn pipeline,
cross-validation helper, or stacking ensemble.

Only classification is covered for now because the Titanic project uses a
binary label.  Extending to regression would require a trivial change:
replace the probability/threshold logic in :meth:`YDFWrapper.predict`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class YDFWrapper(BaseEstimator, ClassifierMixin):
    """scikit-learn adapter for a *pre-configured* YDF classifier.

    The wrapper delegates all heavy lifting to the underlying
    ``ydf.GradientBoostedTreesLearner`` (or any YDF learner that exposes
    ``.train`` and ``.predict``).  Its sole purpose is to implement the
    sklearn estimator contract so you can use functions such as
    :func:`sklearn.model_selection.cross_val_score`,
    :func:`sklearn.ensemble.StackingClassifier`, etc.

    Example
    -------
    ```python
    from titanic.models import YDFWrapper
    import ydf

    learner = ydf.GradientBoostedTreesLearner(label="Survived")
    clf = YDFWrapper(learner).fit(X_train, y_train)
    print(clf.predict(X_test))
    ```

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels.  Set during :meth:`fit`.
    feature_names_in_ : list[str]
        Column names learned during :meth:`fit`.  Used to rebuild
        DataFrames when callers pass a plain numpy array to
        :meth:`predict`/`:py:meth:predict_proba`.
    n_features_in_ : int
        Number of input features seen during training.
    model_ : ydf.Model
        Trained YDF model instance.  Available after :meth:`fit`.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(self, learner: Any):
        """Create a wrapper around an *un-trained* YDF learner.

        Args:
            learner: A configured YDF learner, e.g.
                ``ydf.GradientBoostedTreesLearner(label="Survived")``.
                Must expose ``train`` and ``predict`` methods.
        """
        self.learner = learner

    # ------------------------------------------------------------------
    #  sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "YDFWrapper":  # noqa: N802
        """Fit the underlying YDF model.

        Args:
            X: Feature matrix **as a pandas DataFrame**.  A DataFrame is
               required because YDF needs column names.
            y: Target vector or pandas Series.

        Raises:
            ValueError: If *X* is not a DataFrame.

        Returns:
            self
        """
        if not hasattr(X, "columns"):
            raise ValueError("YDFWrapper.fit expects a pandas DataFrame.")

        self.classes_ = np.unique(y)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        df_train = X.copy()
        df_train["Survived"] = y.values if hasattr(y, "values") else y
        self.model_ = self.learner.train(df_train, verbose=0)
        return self

    def predict(self, X):  # noqa: N802
        """Return hard class predictions (0 or 1).

        Args:
            X: Feature matrix.  Can be a DataFrame *or* numpy array.  If
               an array is provided, it is wrapped into a DataFrame using
               :pyattr:`feature_names_in_`.

        Returns:
            ndarray[int] of shape (n_samples,)
        """
        proba = self._predict_proba_raw(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):  # noqa: N802
        """Return class probabilities in sklearn's ``[P(class=0), P(class=1)]`` format.

        Args:
            X: Feature matrix (DataFrame or ndarray).

        Returns:
            ndarray[float] of shape (n_samples, 2)
        """
        p1 = self._predict_proba_raw(X)
        return np.vstack([1 - p1, p1]).T

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _predict_proba_raw(self, X) -> np.ndarray:
        """Return ``P(class=1)`` as a 1-D array without any thresholding."""
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return self.model_.predict(X)
