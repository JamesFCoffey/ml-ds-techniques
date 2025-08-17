from typing import Any, Mapping, Sequence


def tokenize_pairs(
    tokenizer,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    max_len: int,
    return_tensors: str = "np",
) -> Mapping[str, Any]:
    """Tokenize premise–hypothesis pairs for transformer encoders.

    This helper wraps a Hugging Face tokenizer to produce fixed‑length,
    model‑ready inputs for Natural Language Inference (NLI). It pads/truncates
    both sequences to ``max_len`` and returns tensors/arrays according to
    ``return_tensors``.

    Args:
      tokenizer: A Hugging Face tokenizer (e.g., ``AutoTokenizer`` instance)
        that implements ``__call__`` for pair encoding.
      premises: Sequence of strings containing the premise texts.
      hypotheses: Sequence of strings containing the hypothesis texts.
      max_len: Maximum sequence length to pad/truncate to. Must match the
        input shape used to build the Keras model.
      return_tensors: One of {``"np"``, ``"tf"``, ``"pt"``}. Controls the
        tensor backend of the returned encoding.

    Returns:
      Mapping[str, Any]: A dict mapping feature names to arrays/tensors,
      typically including ``"input_ids"`` and ``"attention_mask"`` and (for
      some models) optional ``"token_type_ids"``. The value types depend on
      ``return_tensors``.

    Raises:
      ValueError: If the input sequences have mismatched lengths.

    Example:
      >>> enc = tokenize_pairs(tok, ["A dog runs."], ["An animal moves."], max_len=128)
      >>> sorted(enc.keys())
      ['attention_mask', 'input_ids']
    """
    return tokenizer(
        list(premises),
        list(hypotheses),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors=return_tensors,
    )
