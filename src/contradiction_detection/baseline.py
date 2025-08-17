import pandas as pd


def join_pair(df: "pd.DataFrame") -> list[str]:
    """Concatenate premise and hypothesis for bag‑of‑words baselines.

    Produces a text input of the form ``"<premise> [SEP] <hypothesis>"`` used by
    TF‑IDF / linear models that do not rely on special tokenization.

    Args:
      df: DataFrame containing ``"premise"`` and ``"hypothesis"`` columns.

    Returns:
      list[str]: Concatenated strings suitable for vectorization.

    Example:
      >>> join_pair(pd.DataFrame({"premise": ["A"], "hypothesis": ["B"]}))
      ['A [SEP] B']
    """
    return (df["premise"] + " [SEP] " + df["hypothesis"]).tolist()
