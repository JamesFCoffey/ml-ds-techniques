"""High-level façade for the *Flower-Classification* helper package.

Importing this top-level module gives you a **curated public surface** that
covers everything the notebook needs:

* **Data-pipeline helpers** – `configure`, `build_dataset`, `augment`, …
* **Model builders**       – `build_model`, `model_builder`, `recompile`
* **Training callbacks**   – `MacroF1`, `WandbCallback`

Anything *not* re-exported through `__all__` should be treated as private
implementation detail.  A typical usage pattern inside the notebook is::

    from flower_classification import (
        configure,
        build_dataset,
        build_model,
        MacroF1,
    )

All sub-modules are lazily imported, so the statement above is inexpensive even
when running on the constrained Kaggle TPU VM.
"""

from .callbacks import MacroF1, WandbCallback
from .data import (
    augment,
    build_dataset,
    configure,
    decode_image,
    parse_test,
    parse_tfrecord,
)
from .model import build_model, model_builder, recompile

__all__ = [
    "augment",
    "build_dataset",
    "build_model",
    "configure",
    "decode_image",
    "MacroF1",
    "model_builder",
    "parse_test",
    "parse_tfrecord",
    "recompile",
    "WandbCallback",
]
