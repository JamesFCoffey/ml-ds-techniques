"""Model-building helpers for the Flower-Classification TPU notebook.

This module isolates all **model architecture** concerns so the Jupyter
notebook can stay focused on experiment orchestration.  Every public helper
is fully agnostic to the caller’s runtime: the same code path works on
CPU/GPU or on an 8-core TPU as long as the appropriate
`tf.distribute.Strategy` is passed in.

Exports
-------
build_model(strategy, img_size, classes, dropout_rate=0.3)
    Returns an un-trained EfficientNet-B3 classifier wrapped in the supplied
    distribution strategy scope.

model_builder(hp, strategy, img_size, classes)
    A `keras_tuner`-compatible factory that samples *dropout* and *learning
    rate* hyper-parameters, builds the network and compiles it in one go.

recompile(model, lr)
    Utility that re-compiles an **already built** model with a new learning
    rate; handy for stage-wise fine-tuning.

Example
-------
```python
from kerastuner.tuners import BayesianOptimization
from flower_classification.model import build_model, model_builder

# Single-shot build -----------------------------------------------------
strategy = tf.distribute.TPUStrategy()
model = build_model(strategy, img_size=299, classes=104)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Hyper-parameter search ------------------------------------------------
tuner = BayesianOptimization(
    hypermodel=lambda hp: model_builder(hp, strategy, 299, 104),
    objective='val_accuracy',
    max_trials=20,
)
tuner.search(train_ds, validation_data=val_ds, epochs=3)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3


def build_model(strategy, img_size, classes, dropout_rate=0.3):
    """Constructs an EfficientNet-B3 classifier with a configurable dropout.

    Args:
        strategy:  An initialized `tf.distribute.Strategy`.
        img_size:  Input resolution (int).
        classes:   Number of output classes.
        dropout_rate: Dropout applied after the global-pooling layer.

    Returns:
        A compiled but **un-trained** `tf.keras.Model`.
    """
    with strategy.scope():
        base = EfficientNetB3(
            include_top=False,
            input_shape=(img_size, img_size, 3),
            weights="imagenet",
            pooling="avg",
        )
        x = layers.Dropout(dropout_rate)(base.output)
        outputs = layers.Dense(classes, activation="softmax")(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        return model


def model_builder(hp, strategy, img_size, classes):
    """`keras_tuner` HyperModel factory.

    Args:
        hp: A `kerastuner.engine.hyperparameters.HyperParameters` instance
            supplying the current sample of hyper-parameters.
        strategy:  An initialized `tf.distribute.Strategy`.
        img_size:  Input resolution (int).
        classes:   Number of output classes.

    Returns:
        A compiled model ready for training inside a tuner trial.
    """
    dr = hp.Float("dropout", 0.2, 0.5, step=0.1)
    lr = hp.Choice("lr", [1e-2, 5e-3, 1e-3])
    model = build_model(strategy, img_size, classes, dropout_rate=dr)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def recompile(model, lr):
    """Re-compiles the global `model` with a new learning rate.

    Args:
        model:  A compiled but `tf.keras.Model`.
        lr: The learning-rate value to feed into `tf.keras.optimizers.Adam`.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
