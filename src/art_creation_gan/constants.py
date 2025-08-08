"""Package‑wide hyper‑parameters and deterministic seeding.

All numeric constants live here so notebooks and modules can `import` rather
than hard‑code:

* **IMG_SIZE** – square image resolution (Monet dataset is 256 × 256).
* **BATCH**    – per‑replica batch size (multiplied by TPU replicas later).
* **BUFFER**   – shuffle buffer for TFRecord streams.
* **AUTO**     – alias for `tf.data.AUTOTUNE`.
* **SEED**     – global RNG seed to keep experiments repeatable.
* **OUTPUT_CHANNELS** – RGB channel count.

Changing a value here updates every dependent module automatically.
"""

import tensorflow as tf

SEED = 42
IMG_SIZE: int = 256  # Height & width of images
BATCH: int = 8  # Per‑replica batch size (TPU v3‑8 ⇒ 64 global)
BUFFER: int = 1024  # Shuffle buffer size for TFRecords
AUTO = tf.data.AUTOTUNE  # Let TF tune prefetch & parallel calls
OUTPUT_CHANNELS = 3  # RGB
