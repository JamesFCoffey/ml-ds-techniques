"""Custom Keras callbacks used by the Kaggle TPU notebook.

The file purposefully keeps the callback logic **stateless** so the objects can
be pickled by `keras_tuner` and distributed across TPU workers without trouble.

Exports:
    * `MacroF1`      – computes a macro-averaged F1 score after every epoch and
                       injects it into `logs` under the key `'val_macro_f1'`.
    * `WandbCallback` – a no-op drop-in replacement whenever Weights & Biases is
                       unavailable on the runtime image.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks


class MacroF1(callbacks.Callback):
    """Computes macro-averaged F1 on a held-out `val_data` dataset.

    The callback is intentionally placed **first** in the training callback
    list so that its `logs['val_macro_f1']` entry is available to later
    callbacks such as `EarlyStopping` or `ModelCheckpoint`.

    Attributes:
        val_data: A `tf.data.Dataset` yielding `(image_batch, one_hot_label)`
            pairs used for validation after every epoch.
    """

    def __init__(self, val_data):
        """Initializes the callback.

        Args:
            val_data: Pre-batched `tf.data.Dataset` used for F1 evaluation.
        """
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        """Runs after each epoch and injects `val_macro_f1` into `logs`.

        Args:
            epoch: Integer epoch index (0-based).
            logs:  Dictionary of metric results provided by Keras. The method
                adds a new key `'val_macro_f1'`.
        """
        y_true, y_pred = [], []
        for imgs, lbls in self.val_data:
            preds = self.model.predict(imgs, verbose=0)
            y_true.extend(np.argmax(lbls.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        f1 = f1_score(y_true, y_pred, average="macro")
        logs = logs or {}
        logs["val_macro_f1"] = f1
        print(f" — val_macro_f1: {f1:.4f}")


class WandbCallback(tf.keras.callbacks.Callback):
    """Dummy replacement when `wandb.keras.WandbCallback` is unavailable.

    This stand-in satisfies the Keras callback interface so downstream code
    (which always appends *some* `WandbCallback` to `callbacks=list`)
    continues to work even when Weights & Biases isn’t installed or when the
    import fails inside Kaggle’s minimal Docker image.

    It purposefully does **nothing** on any callback hook.
    """

    def __init__(self, *args, **kwargs):
        """Creates the no-op callback; all args are ignored."""
        super().__init__()
