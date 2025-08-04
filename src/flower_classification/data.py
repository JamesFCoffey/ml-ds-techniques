"""TFRecord data-pipeline utilities for the Flower-Classification project.

Key design points
-----------------
* **Runtime configuration.**  The first thing the notebook does is call
  :func:`configure`, injecting resolution, batch size, augmentation settings,
  *etc.* into module-level globals.  All other helpers assert that the
  configuration is in place before doing any work.
* **TPU-friendly.**  Every `Dataset` produced by :func:`build_dataset` follows
  Google’s TPU performance recommendations (auto sharding, fused map➜batch,
  deterministic off when shuffling, prefetch on `AUTOTUNE`, and so on).
* **Schema flexibility.**  :func:`parse_tfrecord` gracefully handles both the
  “official” competition shards (*label*, *img*) and the older starter shards
  (*class*, *image*).

Public API
----------
configure           – Stores run-time constants (MUST be called first).
build_dataset       – Factory that yields a fully-prefetched Dataset.
decode_image        – JPEG → float32 tensor, central-crop & resize.
parse_tfrecord      – Generic record parser for train / val shards.
parse_test          – Specialised parser for test shards (no label).
augment             – Lightweight, XLA-compatible augmentation function.
"""

from typing import Any, Dict

import tensorflow as tf

_RAW_RES: int | None = None
_IMAGE_SZ: int | None = None
_IMAGE_FEATURE_DESCRIPTION: Dict[str, tf.io.FixedLenFeature] | None = None
_AUTO: Any = tf.data.AUTOTUNE
_CLASSES: int | None = None
_BATCH: int | None = None
_AUG_CFG: Dict[str, Any] | None = None


def configure(
    *,
    raw_res: int,
    image_sz: int,
    feature_description: Dict[str, tf.io.FixedLenFeature],
    classes: int,
    batch: int,
    aug_cfg: Dict[str, Any],
    auto=tf.data.AUTOTUNE,
) -> None:
    """Registers all run-time constants required by the data pipeline.

    This **must** be the first call you make after importing
    `flower_classification.data`.  It copies parameters coming from the
    notebook (or other driver script) into module-level globals so that every
    helper—`decode_image`, `build_dataset`, `augment`, etc.—can read the exact
    same configuration without argument plumbing.

    Args:
        raw_res: The square side length (in pixels) of the *raw* images stored
            inside the TFRecords. The decoder will center-crop / pad to this
            size **before** resizing to `image_sz`.
        image_sz: Final side length fed to the network, e.g. `299` for
            EfficientNet-B3.  `decode_image()` will always output
            `(image_sz, image_sz, 3)`.
        feature_description: A dict mapping TFRecord feature names to their
            `tf.io.FixedLenFeature` spec.  Passed straight to
            `tf.io.parse_single_example`.
        classes: Total number of flower classes (used for one-hot encoding).
        batch: Global batch size **after** distribution (i.e. the size that
            reaches the model’s `fit()`), not the per-replica size.
        aug_cfg: Dict holding lightweight augmentation hyper-parameters
            (`flip_left_right`, `brightness`, *etc.*) consumed by `augment()`.
        auto: The constant to use for `num_parallel_calls` / `prefetch`
            (defaults to `tf.data.AUTOTUNE`).

    Returns:
        None.  The function is executed purely for its side effects.

    Raises:
        ValueError: If any numeric argument is non-positive.
    """
    if min(raw_res, image_sz, classes, batch) <= 0:
        raise ValueError(
            "`raw_res`, `image_sz`, `classes`, and `batch` must all be > 0."
        )

    global _RAW_RES, _IMAGE_SZ, _IMAGE_FEATURE_DESCRIPTION
    global _CLASSES, _BATCH, _AUG_CFG, _AUTO
    _RAW_RES = raw_res
    _IMAGE_SZ = image_sz
    _IMAGE_FEATURE_DESCRIPTION = feature_description
    _CLASSES = classes
    _BATCH = batch
    _AUG_CFG = aug_cfg
    _AUTO = auto


def _assert_config():
    """Verifies that :func:`configure` has been called.

    All public helpers depend on the module-level constants set by
    :func:`configure`.  Calling any of them beforehand is a logic error that
    would otherwise manifest as obscure `NoneType` failures.  This guard
    converts those latent bugs into an immediate, readable exception.

    Raises:
        RuntimeError: If at least one required global (listed in
            `_REQUIRED_VARS`) is still `None`, meaning `configure()` was never
            invoked.
    """
    _REQUIRED_VARS = [
        "_RAW_RES",
        "_IMAGE_SZ",
        "_IMAGE_FEATURE_DESCRIPTION",
        "_CLASSES",
        "_BATCH",
        "_AUG_CFG",
    ]
    for name in _REQUIRED_VARS:
        if globals().get(name) is None:
            raise RuntimeError(
                "Configuration missing – call `data.configure(...)` before "
                f"using the data API (variable '{name}' is unset)."
            )


def decode_image(image_bytes):
    """Decodes and standardizes a JPEG byte string.

    Args:
        image_bytes: A scalar string `tf.Tensor` holding the raw JPEG bytes.

    Returns:
        A float-tensor image of shape `(IMAGE_SZ, IMAGE_SZ, 3)` in `[0, 1]`.
    """
    _assert_config()
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, _RAW_RES, _RAW_RES)
    img = tf.image.resize(img, [_IMAGE_SZ, _IMAGE_SZ])
    return img


def parse_tfrecord(example_proto):
    """Parses one serialized TFRecord example.

    Handles the slight schema differences between train/val and test shards.

    Args:
        example_proto: A scalar string `tf.Tensor` containing a serialized
            `tf.train.Example`.

    Returns:
        Tuple `(image, label_or_-1, id_str)` where:
          * `image` is a float tensor ready for the model,
          * `label_or_-1` is an `int64` (-1 for test shards),
          * `id_str` is the sample’s unique ID (empty for train/val).
    """
    _assert_config()
    example = tf.io.parse_single_example(example_proto, _IMAGE_FEATURE_DESCRIPTION)
    # pick whichever image key is populated
    img_bytes = tf.cond(
        tf.strings.length(example["img"]) > 0,
        lambda: example["img"],
        lambda: example["image"],
    )
    img = decode_image(img_bytes)

    # choose correct label key ("label" for our spec, "class" for starter shards)
    label_val = tf.where(example["label"] >= 0, example["label"], example["class"])
    id_ = example["id"]
    return img, label_val, id_


def build_dataset(filenames, labeled=True, shuffle=False, repeat=False):
    """Creates a performance-tuned `tf.data.Dataset` for TPU / GPU training.

    Args:
        filenames: List/`tf.Tensor` of TFRecord shard paths.
        labeled:   If `True`, returns `(image, one_hot_label)` batches.
        shuffle:   Whether to `shuffle()` the dataset.
        repeat:    Whether to call `repeat()` to create an endless stream.

    Returns:
        A prefetched, batched `tf.data.Dataset`.
    """
    _assert_config()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.AUTO
    )

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=_AUTO)
    ds = ds.with_options(options)
    if shuffle:
        ds = ds.shuffle(2048)
        # disable deterministic order for speed – TensorFlow ≤2.18 style
        opt_det = tf.data.Options()
        opt_det.experimental_deterministic = False
        ds = ds.with_options(opt_det)

    ds = ds.map(parse_tfrecord, num_parallel_calls=_AUTO)
    if repeat:
        ds = ds.repeat()
    if labeled:
        ds = ds.map(
            lambda x, y, z: (x, tf.one_hot(y, _CLASSES)), num_parallel_calls=_AUTO
        )
    else:
        ds = ds.map(lambda x, y, z: x, num_parallel_calls=_AUTO)
    ds = ds.batch(_BATCH, drop_remainder=labeled)  # keep batches equal for TPU
    ds = ds.prefetch(_AUTO)
    return ds


def parse_test(example_proto):
    """Parses a serialized TFRecord coming from *test* shards.

    The record contains an *id* and the JPEG image bytes but no label.

    Args:
        example_proto: A scalar string `tf.Tensor` with a serialized example.

    Returns:
        Tuple `(image_tensor, id_str)`.
    """
    _assert_config()
    feature_desc = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "img": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    example = tf.io.parse_single_example(example_proto, feature_desc)
    img_bytes = tf.cond(
        tf.strings.length(example["img"]) > 0,
        lambda: example["img"],
        lambda: example["image"],
    )
    img = decode_image(img_bytes)
    return img, example["id"]


@tf.function
def augment(img):
    """Lightweight, XLA-compatible augmentation used during training.

    The ops are all graph-friendly so the function can be decorated with
    `@tf.function`, enabling JIT compilation on TPU.

    Args:
        img: A single image tensor in `[0, 1]`.

    Returns:
        The augmented image tensor (same shape and dtype).
    """
    _assert_config()
    if _AUG_CFG["flip_left_right"]:
        img = tf.image.random_flip_left_right(img)
    if _AUG_CFG["flip_up_down"]:
        img = tf.image.random_flip_up_down(img)
    if _AUG_CFG["brightness"] > 0:
        img = tf.image.random_brightness(img, _AUG_CFG["brightness"])
    if _AUG_CFG["contrast"] > 0:
        img = tf.image.random_contrast(
            img, 1 - _AUG_CFG["contrast"], 1 + _AUG_CFG["contrast"]
        )
    if _AUG_CFG["rotation"] > 0:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
    return img
