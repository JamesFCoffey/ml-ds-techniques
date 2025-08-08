"""High‑level `CycleGan` composite model and custom training loop.

Imports the generator and discriminator factories plus the loss utilities, then
packages them into a single `keras.Model` that supports `compile()` / `fit()`.
The module is intentionally self‑contained so that a user can:

```python
from art_creation_gan.gan import CycleGan
````

and immediately instantiate a train‑ready model without fishing for internal
symbols.
"""

import tensorflow as tf
from tensorflow import keras


class CycleGan(keras.Model):
    """CycleGAN composite model with custom training loop.

    This class bundles two generators and two discriminators into a single
    `keras.Model` so that we can leverage Keras fit/compile semantics while
    retaining full control over the adversarial, cycle‑consistency, and
    identity losses described in the original *Unpaired Image‑to‑Image
    Translation using Cycle‑Consistent Adversarial Networks* paper (Zhu et al.,
    2017).

    During each training step the model performs the following sub‑steps:

    1. **Forward translation** – photo → Monet (fake) and Monet → photo (fake).
    2. **Cycle translation** – translate fakes back to their original domain.
    3. **Identity mapping** – pass real images through their own‑domain
       generator to discourage color shifts.
    4. **Adversarial updates** – compute generator and discriminator losses
       (LSGAN variant) and apply gradients with independent Adam optimizers.

    Attributes:
        m_gen: Generator `keras.Model` mapping *photos ➔ Monet*.
        p_gen: Generator `keras.Model` mapping *Monet ➔ photos*.
        m_disc: Discriminator judging real vs. fake Monet images.
        p_disc: Discriminator judging real vs. fake photo images.
        lambda_cycle: Weight applied to cycle‑consistency and identity losses.
    """

    def __init__(
        self,
        monet_generator: keras.Model,
        photo_generator: keras.Model,
        monet_discriminator: keras.Model,
        photo_discriminator: keras.Model,
        lambda_cycle: int = 10,
    ) -> None:
        """Initializes the composite CycleGAN.

        Args:
            monet_generator: Pre‑built generator that converts photos to the
                Monet style.
            photo_generator: Generator that converts Monet paintings back to
                photo style (inverse mapping).
            monet_discriminator: Discriminator that classifies real vs fake
                Monet images.
            photo_discriminator: Discriminator that classifies real vs fake
                photo images.
            lambda_cycle: Scaling factor for the cycle‑consistency loss term.
        """
        super().__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
        self,
        *,
        m_gen_optimizer: keras.optimizers.Optimizer,
        p_gen_optimizer: keras.optimizers.Optimizer,
        m_disc_optimizer: keras.optimizers.Optimizer,
        p_disc_optimizer: keras.optimizers.Optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn,
    ) -> None:
        """Configures optimizers and loss callables.

        **All arguments must be passed by keyword** to avoid accidental mixing
        of optimizers.

        Args:
            m_gen_optimizer: Optimizer for the photo➔Monet generator.
            p_gen_optimizer: Optimizer for the Monet➔photo generator.
            m_disc_optimizer: Optimizer for the Monet discriminator.
            p_disc_optimizer: Optimizer for the photo discriminator.
            gen_loss_fn: Callable implementing the adversarial generator loss
                (expects discriminator logits for fake images).
            disc_loss_fn: Callable implementing the discriminator loss (expects
                real and fake logits).
            cycle_loss_fn: Callable computing cycle‑consistency L1 loss.
            identity_loss_fn: Callable computing identity‑mapping L1 loss.
        """
        super().compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def call(self, inputs, training: bool = False):
        """Runs a full forward‑and‑cycle pass (no gradient side‑effects).

        This method is primarily used for **inference/visualization**.  It
        translates each domain, cycles back, and returns all intermediate
        tensors so callers can inspect generator outputs.

        Args:
            inputs: Tuple `(real_monet, real_photo)` where each element is a
                batch of images scaled to `[-1, 1]`.
            training: Forwarded to internal layers so things like Dropout or
                GroupNorm behave correctly.

        Returns:
            Tuple `(fake_monet, cycled_photo, fake_photo, cycled_monet)` with
            tensors in the same order as described above.
        """
        real_monet, real_photo = inputs
        fake_monet = self.m_gen(real_photo, training=training)
        cycled_photo = self.p_gen(fake_monet, training=training)
        fake_photo = self.p_gen(real_monet, training=training)
        cycled_monet = self.m_gen(fake_photo, training=training)
        return fake_monet, cycled_photo, fake_photo, cycled_monet

    def train_step(self, batch_data):
        """Executes one training step comprising G & D updates for both domains.

        The method follows the standard Keras `train_step` contract so the
        model can be trained via `model.fit`.  Internally it computes generator
        and discriminator losses, applies gradients with the optimizers
        provided in `compile`, and returns a dictionary of metrics.

        Args:
            batch_data: Tuple `(real_monet, real_photo)` drawn from the zipped
                training dataset.

        Returns:
            A `dict` mapping metric names to scalar tensors — these get logged
            by Keras and shown in the training progress bar.
        """
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # ---------- Forward pass (G) ---------- #
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # ---------- Identity mapping ---------- #
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # ---------- Discriminator logits ------- #
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # ---------- Losses --------------------- #
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            )
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )

            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # ---------- Gradients & optimizer steps ---- #
        self.m_gen_optimizer.apply_gradients(
            zip(
                tape.gradient(total_monet_gen_loss, self.m_gen.trainable_variables),
                self.m_gen.trainable_variables,
            )
        )
        self.p_gen_optimizer.apply_gradients(
            zip(
                tape.gradient(total_photo_gen_loss, self.p_gen.trainable_variables),
                self.p_gen.trainable_variables,
            )
        )
        self.m_disc_optimizer.apply_gradients(
            zip(
                tape.gradient(monet_disc_loss, self.m_disc.trainable_variables),
                self.m_disc.trainable_variables,
            )
        )
        self.p_disc_optimizer.apply_gradients(
            zip(
                tape.gradient(photo_disc_loss, self.p_disc.trainable_variables),
                self.p_disc.trainable_variables,
            )
        )

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss,
        }
