"""Training-time helpers for the Spaceship Titanic project.

This module provides two lightweight utilities that are reused by both
notebooks and command-line scripts:

* **`make_pipe`** – Compose a scikit-learn `Pipeline` that chains a
  `SpaceshipTransformer`, an optional `StandardScaler`, and an arbitrary
  estimator.  The helper hides the boilerplate needed to keep sparse matrices
  sparse when scaling is requested.

* **`make_fold_data`** – Given train/validation row indices produced by a
  cross-validator, fit a fresh `SpaceshipTransformer` on the training slice,
  transform both splits, and return `(X_train, X_valid, y_train, y_valid)`.
  This guarantees that *each* fold sees its own transformer fitted **without
  peeking** at the validation data, preventing leakage.

Both functions are intentionally stateless and quick to unit-test.
"""

from sklearn import pipeline, preprocessing

from .featurize import SpaceshipTransformer


# Build a pipeline with/without scaling
def make_pipe(feat, model, scale_needed=False):
    """Create a scikit-learn pipeline for a tabular classifier.

    Args:
        feat (TransformerMixin): Any fitted-or-fit-capable transformer that
            converts a raw data frame into a numeric matrix (e.g.
            `SpaceshipTransformer`).
        model (BaseEstimator): The final estimator, such as
            `HistGradientBoostingClassifier` or `CatBoostClassifier`.
        scale_needed (bool, optional): If ``True`` the pipeline inserts a
            `StandardScaler(with_mean=False)` **after** the feature transformer.
            Setting `with_mean=False` avoids converting sparse input to dense,
            which would otherwise inflate memory usage.  Defaults to ``False``.

    Returns:
        sklearn.pipeline.Pipeline: Three-step pipeline
        ``[('feat', feat), ('sc', StandardScaler)?, ('clf', model)]`` ready for
        cross-validation or `.fit()`.
    """

    steps = [("feat", feat)]
    if scale_needed:
        # StandardScaler(with_mean=False) keeps sparse → dense explosions away
        steps.append(("sc", preprocessing.StandardScaler(with_mean=False)))
    steps.append(("clf", model))
    return pipeline.Pipeline(steps)


# reuse one transformer instance per fold
def make_fold_data(tr_idx, va_idx, X, y):
    """Fit–transform a single CV fold with leakage-free preprocessing.

    A new `SpaceshipTransformer` is *fitted only* on the training slice
    (``tr_idx``) and then applied to both the training and validation slices.
    The function returns NumPy arrays so that downstream model code can remain
    agnostic to the original pandas index.

    Args:
        tr_idx (array-like): Row indices for the training portion of the fold.
        va_idx (array-like): Row indices for the validation portion.
        X (pd.DataFrame): Raw feature frame containing **all** rows.
        y (pd.Series | np.ndarray): Target vector aligned with ``X``.

    Returns:
        tuple:
            * **X_tr (np.ndarray)** – Transformed training features.
            * **X_va (np.ndarray)** – Transformed validation features.
            * **y_tr (pd.Series | np.ndarray)** – Training labels.
            * **y_va (pd.Series | np.ndarray)** – Validation labels.

    Example:
        >>> for tr_idx, va_idx in cv.split(X, y, groups):
        ...     X_tr, X_va, y_tr, y_va = make_fold_data(tr_idx, va_idx, X, y)
        ...     model.fit(X_tr, y_tr)
        ...     print(model.score(X_va, y_va))
    """

    tfm = SpaceshipTransformer(min_freq=0.01)
    X_tr = tfm.fit_transform(X.iloc[tr_idx])
    X_va = tfm.transform(X.iloc[va_idx])
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    return X_tr, X_va, y_tr, y_va
