"""TFRecord parsing and data‑pipeline utilities.

* `_parse_example()`  – bytes → float tensor in `[-1, 1]`.
* `_augment()`        – light colour/flip crop augmentations.
* `make_dataset()`    – assembles a shuffled, batched, prefetched
  `tf.data.Dataset` ready for TPU consumption.

Centralizing I/O logic ensures dataset handling is identical between training
and inference scripts.
"""

import tensorflow as tf
from .constants import AUTO, BATCH, BUFFER, IMG_SIZE, SEED

# Feature description mirrors Kaggle’s TFRecord schema
_IMAGE_FEATURE_DESCRIPTION = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.string),  # Unused placeholder
}


def _parse_example(proto: tf.Tensor) -> tf.Tensor:
    """Decodes a single TFRecord example into a normalized image tensor.

    Each TFRecord stores one Monet or photo sample under the feature key
    `"image"`.  The raw bytes are decoded from JPEG, cast to `tf.float32`,
    and linearly rescaled from the original `[0, 255]` range to `[-1, 1]`
    so they match the generator’s `tanh` output scale.

    Args:
        proto: A scalar `tf.string` tensor containing the serialized
            `tf.train.Example` pulled from a `TFRecordDataset` iterator.

    Returns:
        A `tf.Tensor` of shape `(256, 256, 3)` with dtype `tf.float32`
        and values in `[-1, 1]`.
    """
    example = tf.io.parse_single_example(proto, _IMAGE_FEATURE_DESCRIPTION)
    img = tf.image.decode_jpeg(example["image"], channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # Rescale to [‑1, 1]
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def _augment(img: tf.Tensor) -> tf.Tensor:
    """Applies lightweight color and spatial augmentations.

    The goal is to diversify the **Monet** domain without breaking its
    characteristic palette or brushwork.  Augmentations are purposefully
    mild compared to large-scale classification pipelines.

    Augmentations performed (in order):
      1. Horizontal flip with 50 % probability.
      2. Random brightness shift in `[-0.2, 0.2]`.
      3. Random contrast scale in `[0.8, 1.2]`.
      4. Random hue shift in `[-0.05, 0.05]`.
      5. Resize to `(286, 286)` followed by a random crop back to
         `(256, 256)` — mirrors the augmentation in the original CycleGAN
         TensorFlow tutorial.

    Args:
        img: A tensor of shape `(256, 256, 3)` in `[-1, 1]`.

    Returns:
        A tensor of identical shape and dtype with the same scaling,
        potentially flipped, color-jittered, and crop-shifted.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    # Slight zoom‑crop a 286→256 patch, mirroring CycleGAN paper.
    img = tf.image.resize(img, [286, 286])
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    return img


def make_dataset(tfrec_files, *, augment: bool = False) -> tf.data.Dataset:
    """Creates a shuffled, batched, prefetched `tf.data.Dataset`.

    Args:
        tfrec_files: List or glob of TFRecord paths.
        augment: Whether to apply `_augment` during the map stage.

    Returns:
        A ready‑to‑iterate `tf.data.Dataset` of `(B, 256, 256, 3)` float images.
    """
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTO)
    ds = ds.shuffle(BUFFER, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_example, num_parallel_calls=AUTO)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds
