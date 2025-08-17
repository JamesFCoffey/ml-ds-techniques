from typing import Dict, Iterable, List, Optional, Tuple

import tensorflow as tf


def make_ds(
    enc: Dict[str, "tf.Tensor"],
    y: Optional[Iterable[int]] = None,
    *,
    batch_size: int = 32,
    train: bool = True,
    num_classes: int = 3,
    seed: int = 42,
    prefetch_buffer=tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, List[str]]:
    """Build a `tf.data.Dataset` from tokenized inputs (and labels).

    Creates a dataset of feature dictionaries suitable for feeding a compiled
    Keras model. If labels ``y`` are provided (integer class IDs), they are
    converted to one‑hot vectors of size ``num_classes`` so that
    ``CategoricalCrossentropy(from_logits=True, label_smoothing=...)`` can be
    used consistently.

    Args:
      enc: Mapping from feature names to arrays/tensors (e.g., output of
        :func:`tokenize_pairs`). Recognized keys are ``"input_ids"``,
        ``"attention_mask"``, and optional ``"token_type_ids"``.
      y: Optional 1D iterable of integer class labels with shape
        ``(num_examples,)``. If ``None``, an unlabeled dataset is created.
      batch_size: Batch size for the returned dataset.
      train: If ``True``, the labeled dataset is shuffled with a fixed seed for
        reproducibility. Ignored when ``y`` is ``None``.
      num_classes: Number of output classes for one‑hot encoding.
      seed: RNG seed used for shuffling when ``train`` is ``True``.
      prefetch_buffer: Value passed to ``Dataset.prefetch`` (e.g., ``AUTO``).

    Returns:
      Tuple[tf.data.Dataset, List[str]]: A batched & prefetched dataset and the
      ordered list of feature keys used to build model inputs.

    Raises:
      ValueError: If none of the expected feature keys are present in ``enc``.

    Example:
      >>> ds, keys = make_ds(enc, y=[0, 2, 1], batch_size=2, num_classes=3)
      >>> next(iter(ds.take(1)))[0].keys() == set(keys)
      True
    """
    feature_keys = [
        k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc
    ]
    x = {k: enc[k] for k in feature_keys}
    if y is not None:
        # One-hot labels so we can use CategoricalCrossentropy with label_smoothing
        y_arr = tf.convert_to_tensor(y, dtype=tf.int32)
        y_arr = tf.one_hot(y_arr, depth=num_classes)
        ds = tf.data.Dataset.from_tensor_slices((x, y_arr))
        if train:
            ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices(x)
    return ds.batch(batch_size).prefetch(prefetch_buffer), feature_keys
