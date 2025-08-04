"""digit_recognizer public API.

This *package‑level* module re‑exports the most important helpers so that
end‑users can simply write::

    from digit_recognizer import build_model, run_bayes_tuner

without needing to know the internal folder structure.

The intent is to keep the public surface small and explicit.  Anything **not**
listed in :data:`__all__` should be considered a private implementation detail
subject to change.

Attributes:
    build_model: Callable[..., tf.keras.Model]
        Thin wrapper around :func:`digit_recognizer.train.build_model` that
        constructs and compiles a convolutional network for digit
        classification.
    run_bayes_tuner: Callable[..., Tuple[tf.keras.Model,
        keras_tuner.HyperParameters]]
        Convenience function that runs a Bayesian hyper‑parameter search and
        returns the best‑performing model plus its hyper‑parameters.
"""

from .train import (
    build_model,
    run_bayes_tuner,
)

__all__ = [
    "build_model",
    "run_bayes_tuner",
]
