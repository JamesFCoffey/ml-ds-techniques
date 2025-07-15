"""
This top-level package gathers **re-usable building-blocks** developed for
the "Titanic – Machine Learning from Disaster" competition:

* **Feature engineering** – see :pyfunc:`titanic.featurize.fit_preprocessor`
  and its helpers.
* **Model wrappers** – :pyclass:`titanic.models.YDFWrapper` bridges YDF to
  scikit-learn.
* **Hyper-parameter optimization** – Optuna callbacks / objectives in
  :pymod:`titanic.hyperopt`.

Only the public API needed by notebooks and scripts is re-exported via
``__all__`` below; all other symbols remain internal.
"""

from .featurize import (
    fit_preprocessor,
    transform_preprocessor,
    preprocess_fold,
    PreprocXGB,
    PreprocYDF,
)
from .models import YDFWrapper
from .hyperopt import (
    make_convergence_callback,
    make_gbts_cv_objective,
    make_xgb_cv_objective,
)

__all__ = [
    "fit_preprocessor",
    "transform_preprocessor",
    "preprocess_fold",
    "PreprocXGB",
    "PreprocYDF",
    "YDFWrapper",
    "make_convergence_callback",
    "make_gbts_cv_objective",
    "make_xgb_cv_objective",
]
