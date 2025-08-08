"""Factory functions for CycleGAN generator and discriminator architectures.

* **build_generator()** – ResNet‑9 encoder–decoder with skip connections.
* **build_discriminator()** – 70 × 70 PatchGAN classifier.

Both functions rely on `blocks.downsample` / `blocks.upsample` so that any change
to the low‑level blocks automatically propagates to both models.
"""

import tensorflow as tf
from blocks import downsample, upsample
from constants import IMG_SIZE, OUTPUT_CHANNELS
from tensorflow import keras
from tensorflow.keras import layers


def build_generator() -> keras.Model:
    """Constructs a ResNet‑9 generator (CycleGAN style).

    The architecture mirrors the one proposed in the original CycleGAN paper:

    * **Encoder** – Seven stride‑2 convolution blocks that progressively halve
      spatial resolution (256 → 1) while expanding channel depth.
    * **Decoder** – Seven transpose‑convolution blocks that restore the
      resolution back to 256 × 256 and include skip connections to the encoder
      (U‑Net flavour) to better preserve low‑frequency structure.
    * **Activation** – A final `tanh` layer maps logits to the `[-1, 1]` range
      expected by the loss functions.

    Returns:
        keras.Model: Functional model that maps a batch of RGB images
        (`shape=(B, 256, 256, 3)`, scaled to `[-1, 1]`) to equally‑sized
        stylized images in the same range.
    """
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    # Encoder (downsample): size halves each step.
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),  # (128×128)
        downsample(128, 4),  # (64×64)
        downsample(256, 4),  # (32×32)
        downsample(512, 4),  # (16×16)
        downsample(512, 4),  # (8×8)
        downsample(512, 4),  # (4×4)
        downsample(512, 4),  # (2×2)
        downsample(512, 4),  # (1×1)
    ]

    # Decoder (upsample): mirrors the encoder.
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (2×2)
        upsample(512, 4, apply_dropout=True),  # (4×4)
        upsample(512, 4, apply_dropout=True),  # (8×8)
        upsample(512, 4),  # (16×16)
        upsample(256, 4),  # (32×32)
        upsample(128, 4),  # (64×64)
        upsample(64, 4),  # (128×128)
    ]

    last = layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
        activation="tanh",
    )  # (256×256×3)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])  # Skip the innermost layer in U‑Net fashion

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    outputs = last(x)
    return keras.Model(inputs, outputs, name="generator")


def build_discriminator() -> keras.Model:
    """Creates a 70 × 70 PatchGAN discriminator.

    The network classifies overlapping 70 × 70 patches of an input image as
    *real* or *fake*, yielding a `(B, 30, 30, 1)` map of logits. PatchGANs focus
    on high‑frequency texture, which encourages generators to produce locally
    consistent brush‑strokes while remaining lightweight.

    Returns:
        keras.Model: A discriminator that accepts images of shape
        `(B, 256, 256, 3)` scaled to `[-1, 1]` and outputs patch‑level logits in
        the same batch order.
    """
    init = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3], name="input_image")
    x = inp

    x = downsample(64, 4, apply_instancenorm=False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1, gamma_initializer=gamma_init)(x)
    x = layers.LeakyReLU()(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(x)

    return keras.Model(inp, x, name="discriminator")
