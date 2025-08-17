from transformers import TFAutoModelForSequenceClassification


def load_tf_sequence_classifier(model_ckpt, num_labels=3):
    """Load a TF sequence‑classification model with a PyTorch fallback.

    Attempts to load a ``TFAutoModelForSequenceClassification`` for the given
    checkpoint. If native TensorFlow weights are unavailable, retries with
    ``from_pt=True`` to convert from PyTorch weights on the fly.

    Args:
      model_ckpt: Model checkpoint name or local path understood by
        ``transformers`` (e.g., ``"microsoft/mdeberta-v3-base"``).
      num_labels: Number of target classes for the classification head.

    Returns:
      TFAutoModelForSequenceClassification: The loaded model instance.

    Raises:
      EnvironmentError: If both the TF and PyTorch loading paths fail.

    Example:
      >>> model = load_tf_sequence_classifier("bert-base-multilingual-cased", num_labels=3)
      >>> hasattr(model, "call")
      True
    """
    try:
        return TFAutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels
        )
    except Exception:
        print(
            f"[Info] Native TF weights not found for {model_ckpt}. Trying from_pt=True …"
        )
        return TFAutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, from_pt=True
        )
