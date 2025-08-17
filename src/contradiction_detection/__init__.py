"""Top-level package for **contradiction_detection**.

This package contains small, reusable helpers for working with the
*Contradictory, My Dear Watson* NLI dataset using TensorFlow and
Hugging Face Transformers. The modules are intentionally lightweight and
stateless so they can be imported into notebooks or scripts.

Public API:
  * :func:`tokenize_pairs` – Tokenize premise–hypothesis pairs.
  * :func:`make_ds` – Build `tf.data.Dataset` objects from tokenized features.
  * :func:`load_tf_sequence_classifier` – Load a TF sequence classifier,
    with an automatic PyTorch fallback.
  * :func:`join_pair` – Concatenate text pairs for bag‑of‑words baselines.
  * :func:`predict_pair_stateless` – Stateless single‑example inference.

Example:
    >>> from contradiction_detection import tokenize_pairs, make_ds
    >>> enc = tokenize_pairs(tok, ["A"], ["B"], max_len=128)
    >>> ds, keys = make_ds(enc, y=[0], batch_size=1)

The submodules are importable individually as well, e.g.::

    from contradiction_detection.tokenization import tokenize_pairs

"""

from .baseline import join_pair
from .data import make_ds
from .infer import predict_pair_stateless
from .modeling import load_tf_sequence_classifier
from .tokenization import tokenize_pairs

__all__ = [
    "tokenize_pairs",
    "make_ds",
    "load_tf_sequence_classifier",
    "join_pair",
    "predict_pair_stateless",
]
