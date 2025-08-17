from typing import Sequence

import numpy as np
import tensorflow as tf


def predict_pair_stateless(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    *,
    max_len: int,
    feature_keys: Sequence[str],
) -> int:
    """Run a single NLI prediction without relying on globals.

    The function pads/truncates inputs to ``max_len`` and forwards only the
    feature keys the compiled model expects (e.g., XLM‑R does not use
    ``token_type_ids``). It is intentionally stateless: callers pass the model,
    tokenizer, and the exact input feature keys to use.

    Args:
      model: A compiled Keras model that accepts a dict of tokenized features
        and outputs logits of shape ``(batch, num_classes)``.
      tokenizer: A Hugging Face tokenizer compatible with the model.
      premise: The premise sentence.
      hypothesis: The hypothesis sentence.
      max_len: Maximum sequence length for padding/truncation.
      feature_keys: Iterable of feature names (e.g., ``("input_ids",
        "attention_mask")``) that the model expects.

    Returns:
      int: Predicted class ID where ``0=entailment``, ``1=neutral``,
      ``2=contradiction``.

    Raises:
      KeyError: If a required feature key is missing from the tokenizer output.

    Example:
      >>> k = ("input_ids", "attention_mask")
      >>> predict_pair_stateless(model, tok, "A dog runs.", "An animal moves.", max_len=128, feature_keys=k) in {0,1,2}
      True
    """
    enc = tokenizer(
        premise,
        hypothesis,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf",
    )
    feed = {k: enc[k] for k in feature_keys}
    logits = tf.cast(model(feed, training=False), tf.float32).numpy()[0]
    return int(np.argmax(logits))
