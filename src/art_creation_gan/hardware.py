"""Device‑selection helpers for TPU‑first workflows.

Exposes `detect_tpu()`, which attempts:

1. TPU‑VM (`tf.distribute.TPUStrategy()`)
2. Remote TPU resolver (legacy pods)
3. CPU/GPU mirror fallback

The resulting `tf.distribute.Strategy` lets the rest of the code remain agnostic
to the underlying accelerator.
"""

import tensorflow as tf


def detect_tpu() -> tf.distribute.Strategy:
    """Detects available TPU and returns an appropriate distribution strategy.

    The function attempts TPU‑VM first (hosted notebooks), falls back to a remote
    TPU resolver, and finally defaults to `MirroredStrategy` for CPU/GPU.  This
    abstraction means the rest of the code can use the same `strategy` object
    regardless of hardware.

    Returns:
        A ready‑to‑use `tf.distribute.Strategy`.
    """
    try:
        strategy = tf.distribute.TPUStrategy()  # TPU‑VM path (≈ 1 line)
        print("TPU‑VM detected – replicas:", strategy.num_replicas_in_sync)
        return strategy
    except (ValueError, NotImplementedError):
        # Fall back to remote‑TPU (legacy notebooks / training pods).
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver("local")
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print("Remote TPU detected – replicas:", strategy.num_replicas_in_sync)
            return strategy
        except Exception as e:
            print("TPU not available (", e, ") – falling back to CPU/GPU")
            return tf.distribute.MirroredStrategy()
