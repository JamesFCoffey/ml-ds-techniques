"""Public export surface for the **art_creation_gan** package.

This package wraps a lightweight CycleGAN implementation that targets Kaggle’s
*I’m Something of a Painter Myself* competition.  Its sub‑modules cover:

* **constants**  – project‑wide hyper‑parameters (image size, batch, etc.).
* **hardware**   – utility for detecting a TPU‑VM / remote TPU / GPU fallback.
* **io**         – TFRecord parsing, on‑the‑fly augmentation, `tf.data` helpers.
* **blocks**     – down‑/up‑sampling building blocks shared by generators and
  discriminators.
* **models**     – generator / discriminator factory functions.
* **losses**     – Least‑Squares GAN + cycle & identity loss utilities.
* **gan**        – `CycleGan` composite model with a custom `train_step`.

The symbols re‑exported via `__all__` are the stable, user‑facing API; internal
helper functions remain encapsulated within their respective modules.
"""

# Constants
from .constants import AUTO, SEED

# Composite model
from .gan import CycleGan

# Hardware helpers
from .hardware import detect_tpu

# Data I/O
from .io import make_dataset

# Losses
from .losses import (
    calc_cycle_loss,
    discriminator_loss,
    generator_loss,
    identity_loss,
)

# Building models
from .models import build_discriminator, build_generator

__all__ = [
    "AUTO",
    "SEED",
    "detect_tpu",
    "make_dataset",
    "build_generator",
    "build_discriminator",
    "discriminator_loss",
    "generator_loss",
    "calc_cycle_loss",
    "identity_loss",
    "CycleGan",
]
