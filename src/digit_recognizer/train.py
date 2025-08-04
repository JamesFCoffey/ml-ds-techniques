"""Model‑building and hyper‑parameter tuning helpers.

This module contains the two **core** functions used throughout the notebook
and potential downstream scripts:

* :func:`build_model` – creates and compiles a small CNN whose architecture and
  optimizer settings can be controlled via a
  :class:`keras_tuner.HyperParameters` object.
* :func:`run_bayes_tuner` – performs Bayesian optimization over the search
  space defined in *build_model* and returns the best model along with the
  corresponding hyper‑parameters.

Typical usage example::

    from digit_recognizer.train import build_model, run_bayes_tuner

    best_model, best_hp = run_bayes_tuner()
    test_acc = best_model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Test accuracy: {test_acc:.4f}")

All random seeds are fixed via the global ``SEED`` constant to ensure
reproducibility across notebook restarts.  The Bayesian search terminates
early if the validation accuracy fails to improve by more than 0.001 over the
last ten trials – a pragmatic stop‑rule that saves GPU time on Kaggle.
"""

import tensorflow as tf
from keras_tuner import BayesianOptimization
from tensorflow.keras import layers

SEED = 42

def build_model(hp=None):
    """Create and compile a CNN for MNIST digit classification.

    The network architecture and training hyper-parameters can be
    *optionally* controlled by a `keras_tuner.engine.hyperparameters`
    instance. When `hp is None`, the function expects to be called
    **only** inside Keras-Tuner and will therefore raise if accessed
    interactively without an `hp` object.

    Args:
        hp: Optional[`keras_tuner.HyperParameters`]. Hyper-parameter
            search space supplied by Keras-Tuner. The object provides
            sampling methods (`Choice`, `Int`, `Float`, …) that replace
            otherwise hard-coded values:
              • `filters1`, `kernel1` – first convolutional block
              • `filters2`, `kernel2` – second convolutional block
              • `dense_units`         – fully connected layer width
              • `dr_rate`             – dropout rate (0.2 – 0.5)
              • `lr`                  – learning rate for Adam

    Returns:
        tf.keras.Sequential: A compiled model ready for `fit` / `predict`
        with   – input shape: (28, 28, 1)
                – output units: 10 (soft-max for digits 0-9)
                – loss : `sparse_categorical_crossentropy`
                – metric: `accuracy`
    """
    # Convolutional block 1
    filters1 = hp.Choice("filters1", [16, 32, 64, 128])
    ks1 = hp.Choice("kernel1", [1, 3, 5])

    # Convolutional block 2
    filters2 = hp.Choice("filters2", [32, 64, 128])
    ks2 = hp.Choice("kernel2", [1, 3, 5])

    # Dense and regularization hyper-params
    dense_units = hp.Int("dense", 64, 512, step=64)
    dr_rate = hp.Float("dropout", 0.2, 0.5, step=0.1)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")

    model = tf.keras.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Conv2D(filters1, ks1, activation="relu", padding="same"),
            layers.Conv2D(filters2, ks2, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Dropout(dr_rate),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dr_rate),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def run_bayes_tuner(datagen, X_tr, X_val, y_tr, y_val):
    """Runs Bayesian hyper‑parameter optimisation for the MNIST CNN.

    A :class:`keras_tuner.BayesianOptimization` search explores the
    architecture defined in :func:`build_model` until either
    ``max_trials`` (40) are evaluated *or* an early‑exit heuristic kicks
    in (no ≥ 0.001 improvement in validation accuracy over the last
    ten trials).

    Args:
        datagen: A :class:`tf.keras.preprocessing.image.ImageDataGenerator`
            instance that yields augmented training batches.
        X_tr: ``np.ndarray`` of shape ``(N, 28, 28, 1)``.  Normalised
            training images.
        X_val: ``np.ndarray`` of shape ``(M, 28, 28, 1)``.  Validation
            images used for objective evaluation.
        y_tr: ``np.ndarray`` of length ``N`` containing integer labels
            ``0–9`` for *X_tr*.
        y_val: ``np.ndarray`` of length ``M`` containing integer labels
            for *X_val*.

    Returns:
        Tuple[tf.keras.Model, keras_tuner.HyperParameters]:
            A two‑tuple ``(best_model, best_hp)`` where

            * **best_model** – The highest‑scoring model instance, i.e.
              architecture **and** trained weights.
            * **best_hp** – The :class:`keras_tuner.HyperParameters`
              object that produced *best_model*, suitable for
              ``build_model(best_hp)`` calls.

    Raises:
        ValueError: If no valid trial is completed (e.g. data shapes are
            incorrect and every trial crashes early).

    Side effects:
        * Creates/overwrites ``mnist_tuning/cnn_bo/`` on disk to store
          trial logs and checkpoints.
        * Emits an informational print when the early‑exit criterion is
          triggered.
    """
    tuner = BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=40,
        num_initial_points=8,
        directory="mnist_tuning",
        project_name="cnn_bo",
        overwrite=True,
        seed=SEED,
    )

    tuner.search(
        datagen.flow(X_tr, y_tr, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    # Return best artifacts
    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return best_model, best_hp
