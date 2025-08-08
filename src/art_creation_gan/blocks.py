"""Building‑block layers for CycleGAN generators and discriminators.

Defines two convenience functions—`downsample()` and `upsample()`—that return
pre‑configured `keras.Sequential` blocks:

* **downsample**: `Conv2D → (InstanceNorm) → LeakyReLU`, stride 2.
* **upsample**  : `Conv2DTranspose → InstanceNorm → (Dropout) → ReLU`, stride 2.

Keeping these micro‑architectures in one place avoids copy‑pasta across the
`models` and `gan` modules and ensures layer initializers / normalization styles
stay consistent.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def downsample(
    filters: int, size: int, *, apply_instancenorm: bool = True
) -> keras.Sequential:
    """Creates a down-sampling block used in the encoder path.

    The block performs **Conv ⇒ (InstanceNorm) ⇒ LeakyReLU** with a stride of 2,
    halving the spatial resolution while increasing the channel depth.

    Args:
        filters: Number of convolution filters to apply.
        size: Side length of the square convolution kernel (e.g. 4 for a 4×4
            kernel).
        apply_instancenorm: If `True`, inserts `GroupNormalization` with
            `groups = channels` (i.e. InstanceNorm).  Set to `False` for the
            very first layer, mirroring common CycleGAN practice.

    Returns:
        A `keras.Sequential` block that maps tensors of shape
        `(B, H, W, C)` → `(B, H/2, W/2, filters)`.
    """
    init = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    block = keras.Sequential()
    block.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=init,
            use_bias=False,
        )
    )
    if apply_instancenorm:
        block.add(layers.GroupNormalization(groups=-1, gamma_initializer=gamma_init))
    block.add(layers.LeakyReLU())
    return block


def upsample(
    filters: int, size: int, *, apply_dropout: bool = False
) -> keras.Sequential:
    """Creates an up-sampling block used in the decoder path.

    The block performs **Transposed Conv ⇒ InstanceNorm ⇒ (Dropout) ⇒ ReLU**
    with a stride of 2, doubling the spatial resolution and optionally applying
    dropout (useful for introducing stochasticity near bottleneck layers).

    Args:
        filters: Number of transposed-convolution filters to apply.
        size: Side length of the square transposed-convolution kernel
            (e.g. 4 for a 4×4 kernel).
        apply_dropout: If `True`, inserts a `Dropout(0.5)` layer after
            InstanceNorm.  Recommended for the three innermost decoder layers
            as in the original U-Net and CycleGAN papers.

    Returns:
        A `keras.Sequential` block that maps tensors of shape
        `(B, H, W, C)` → `(B, 2 H, 2 W, filters)`.
    """
    init = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    block = keras.Sequential()
    block.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=init,
            use_bias=False,
        )
    )
    block.add(layers.GroupNormalization(groups=-1, gamma_initializer=gamma_init))
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block
