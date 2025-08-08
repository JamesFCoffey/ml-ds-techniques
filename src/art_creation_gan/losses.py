"""Loss functions for Least‑Squares CycleGAN training.

Implements four stateless helpers:

* **discriminator_loss** – LS‑GAN objective for real vs fake logits.
* **generator_loss**     – adversarial loss encouraging fake logits → 1.
* **calc_cycle_loss**    – L1 reconstruction error weighted by *λ*.
* **identity_loss**      – optional identity mapping term to stabilize colour.

Keeping these in one module avoids circular imports between `models` and `gan`.
"""

import tensorflow as tf


def discriminator_loss(real, generated):
    """Calculates the Least‑Squares GAN loss for the discriminator.

    The discriminator is trained to output values close to **1** for real
    images and **0** for generated (fake) images. The loss is the average of
    two binary cross‑entropy terms, encouraging correct classification of both
    real and fake batches.

    Args:
        real: Tensor of discriminator logits for real images.
        generated: Tensor of discriminator logits for generated images.

    Returns:
        A scalar tensor containing the discriminator loss for the batch.
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(real), real)
    gen_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.zeros_like(generated), generated)
    return 0.5 * (real_loss + gen_loss)


def generator_loss(generated):
    """Computes the generator's adversarial loss.

    The generator aims to fool the discriminator, so it is rewarded when the
    discriminator predicts **1** (real) for generated images. The loss is the
    binary cross‑entropy between the discriminator logits for generated images
    and a target tensor of ones.

    Args:
        generated: Tensor of discriminator logits for generated images.

    Returns:
        A scalar tensor representing the generator loss.
    """
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, lam):
    """Measures cycle‑consistency error between original and cycled images.

    After translating an image to the opposite domain and back again, the output
    should closely match the original. This function computes the mean absolute
    error (L1 distance) between `real_image` and `cycled_image`, then scales it
    by `lam`.

    Args:
        real_image: Batch of source‑domain images.
        cycled_image: Images obtained after forward‑and‑back translation.
        lam: Weighting factor for the cycle‑loss term.

    Returns:
        A scalar tensor with the weighted cycle‑consistency loss.
    """
    return lam * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image, lam):
    """Enforces identity mapping for images already in the target domain.

    Passing a target‑domain image through the generator should ideally leave it
    unchanged. This loss penalizes differences between `real_image` and the
    generator output `same_image`.

    Args:
        real_image: Images that already belong to the generator's target style.
        same_image: Generator output for `real_image`.
        lam: Scaling factor, typically half the cycle‑loss weight.

    Returns:
        A scalar tensor containing the identity‑mapping loss.
    """
    return 0.5 * lam * tf.reduce_mean(tf.abs(real_image - same_image))
