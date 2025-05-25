# src/house_prices/train.py

"""
Define Optuna objective‐function factories for the House Prices regression
challenge.

This module provides factory functions that take preprocessed training and
validation data and return Optuna‐compatible objectives for:

  - CART (single decision tree)
  - Random Forest
  - Gradient Boosted Trees (YDF)
  - XGBoost
  - Multi‐Layer Perceptron (Keras)

Each returned objective accepts a single `trial` argument and returns the
validation RMSE to be minimized.
"""

import math

import xgboost as xgb
import ydf
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.optimizers.schedules import (
    CosineDecay,
    ExponentialDecay,
    InverseTimeDecay,
    PiecewiseConstantDecay,
)


def make_cart_objective(df_train_ydf, df_valid_ydf, y_valid):
    """
    Build an Optuna objective for tuning a single CART model.

    Args:
        df_train_ydf (pd.DataFrame):
            Training features and log‐transformed 'SalePrice' for YDF.
        df_valid_ydf (pd.DataFrame):
            Validation features and log‐transformed 'SalePrice' for YDF.
        y_valid (pd.Series):
            True log‐transformed target values for the validation set.

    Returns:
        Callable[[optuna.trial.Trial], float]:
            A function that Optuna will call; it trains a CART on the training
            data, predicts on validation data, and returns RMSE.
    """

    def objective(trial):
        max_depth = trial.suggest_categorical("max_depth", [3, 5, 10, 15, None])
        min_examples = trial.suggest_categorical("min_examples", [1, 5, 10, 20])
        model = ydf.CartLearner(
            label="SalePrice",
            task=ydf.Task.REGRESSION,
            max_depth=max_depth,
            min_examples=min_examples,
        ).train(df_train_ydf)
        preds = model.predict(df_valid_ydf)
        return mean_squared_error(y_valid, preds, squared=False)

    return objective


def make_rf_objective(df_train_ydf, df_valid_ydf, y_valid):
    """
    Build an Optuna objective for tuning a Random Forest model.

    Args:
        df_train_ydf (pd.DataFrame):
            Training features and log‐transformed 'SalePrice' for YDF.
        df_valid_ydf (pd.DataFrame):
            Validation features and log‐transformed 'SalePrice'.
        y_valid (pd.Series):
            True log‐transformed target values for the validation set.

    Returns:
        Callable[[optuna.trial.Trial], float]:
            Objective that trains a YDF RandomForestLearner and returns
            validation RMSE.
    """

    def objective(trial):
        params = {
            "num_trees": trial.suggest_categorical("num_trees", [100, 500, 1000]),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "num_candidate_attributes_ratio": trial.suggest_categorical(
                "num_candidate_attributes_ratio", [0.3, 0.5, 0.7, 1.0]
            ),
            "min_examples": trial.suggest_categorical("min_examples", [1, 5, 10, 20]),
        }
        model = ydf.RandomForestLearner(
            label="SalePrice",
            task=ydf.Task.REGRESSION,
            compute_oob_performances=True,
            **params,
        ).train(df_train_ydf)
        preds = model.predict(df_valid_ydf)
        return mean_squared_error(y_valid, preds, squared=False)

    return objective


def make_gbts_objective(df_train_ydf, df_valid_ydf, y_valid):
    """
    Build an Optuna objective for tuning a YDF Gradient Boosted Trees model.

    Args:
        df_train_ydf (pd.DataFrame):
            Training features and log‐transformed 'SalePrice' for YDF.
        df_valid_ydf (pd.DataFrame):
            Validation features and log‐transformed 'SalePrice'.
        y_valid (pd.Series):
            True log‐transformed target values for the validation set.

    Returns:
        Callable[[optuna.trial.Trial], float]:
            Objective that trains a YDF GBT and returns validation RMSE.
    """

    def objective(trial):
        params = {
            "num_trees": trial.suggest_int("num_trees", 100, 1000),
            "shrinkage": trial.suggest_float("shrinkage", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "num_candidate_attributes_ratio": trial.suggest_float(
                "num_candidate_attributes_ratio", 0.3, 1.0
            ),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
        }
        model = ydf.GradientBoostedTreesLearner(
            label="SalePrice", task=ydf.Task.REGRESSION, **params
        ).train(df_train_ydf)
        preds = model.predict(df_valid_ydf)
        return mean_squared_error(y_valid, preds, squared=False)

    return objective


def make_xgb_objective(X_train, X_valid, y_train, y_valid):
    """
    Build an Optuna objective for tuning an XGBoost regressor.

    Args:
        X_train (np.ndarray or pd.DataFrame):
            Preprocessed numeric array for training.
        X_valid (np.ndarray or pd.DataFrame):
            Preprocessed numeric array for validation.
        y_train (pd.Series or np.ndarray):
            Log‐transformed target values for training.
        y_valid (pd.Series or np.ndarray):
            Log‐transformed target values for validation.

    Returns:
        Callable[[optuna.trial.Trial], float]:
            Objective that trains an XGBRegressor and returns validation RMSE.
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        # Query the build info
        build_info = xgb.build_info()
        gpu_supported = build_info.get("USE_CUDA", False)

        # Choose tree_method/predictor based on gpu availability
        if gpu_supported:
            tree_method = "gpu_hist"
            predictor = "gpu_predictor"
        else:
            tree_method = "hist"
            predictor = "auto"

        model = xgb.XGBRegressor(
            **params,
            tree_method=tree_method,
            predictor=predictor,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds, squared=False)

    return objective


def make_mlp_objective(X_train_mlp, X_valid_mlp, y_train, y_valid):
    """
    Build an Optuna objective for tuning a Keras MLP regressor.

    Args:
        X_train_mlp (np.ndarray):
            Scaled numeric array for MLP training.
        X_valid_mlp (np.ndarray):
            Scaled numeric array for MLP validation.
        y_train (pd.Series or np.ndarray):
            Log‐transformed target values for training.
        y_valid (pd.Series or np.ndarray):
            Log‐transformed target values for validation.

    Returns:
        Callable[[optuna.trial.Trial], float]:
            Objective that builds, compiles, trains a Keras Sequential MLP and
            returns validation RMSE after early stopping and pruning.
    """

    def objective(trial):
        # Architecture
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = [
            trial.suggest_categorical(
                f"units_l{i}", [8, 16, 32, 64, 128, 256, 512, 1024]
            )
            for i in range(n_layers)
        ]
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "prelu", "elu"]
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        use_bn = trial.suggest_categorical("batch_norm", [True, False])

        # Optimization
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        lr_schedule = trial.suggest_categorical(
            "lr_schedule",
            [
                "constant",
                "time_decay",
                "step_decay",
                "exponential_decay",
                "cosine_annealing",
                "reduce_on_plateau",
            ],
        )

        # Training control
        patience = trial.suggest_int("patience", 3, 6)
        max_epochs = trial.suggest_categorical("max_epochs", [100, 200, 500])

        # Build model
        model = keras.Sequential()
        model.add(layers.InputLayer(shape=(X_train_mlp.shape[1],)))
        for u in units:
            model.add(layers.Dense(u, kernel_regularizer=regularizers.l2(weight_decay)))
            if use_bn:
                model.add(layers.BatchNormalization())
            if activation == "relu":
                model.add(layers.Activation("relu"))
            elif activation == "leaky_relu":
                model.add(layers.LeakyReLU(negative_slope=0.01))
            elif activation == "prelu":
                model.add(layers.PReLU())
            else:
                model.add(layers.ELU())
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))

        # Learning-rate schedule
        if lr_schedule == "constant":
            schedule = lr
        elif lr_schedule == "time_decay":
            schedule = InverseTimeDecay(lr, decay_steps=1000, decay_rate=1.0)
        elif lr_schedule == "step_decay":
            schedule = PiecewiseConstantDecay(
                boundaries=[1000, 2000], values=[lr, lr * 0.1, lr * 0.01]
            )
        elif lr_schedule == "exponential_decay":
            schedule = ExponentialDecay(lr, decay_steps=1000, decay_rate=0.1)
        elif lr_schedule == "cosine_annealing":
            schedule = CosineDecay(lr, decay_steps=max_epochs)
        else:  # reduce_on_plateau
            schedule = lr

        # Optimizer
        if opt_name == "adam":
            optimizer = Adam(learning_rate=schedule, clipnorm=1.0)
        elif opt_name == "sgd":
            optimizer = SGD(learning_rate=schedule, momentum=0.9)
        else:
            optimizer = RMSprop(learning_rate=schedule)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Callbacks: early stopping, pruning, optional ReduceLROnPlateau
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, "val_loss"),
        ]
        if lr_schedule == "reduce_on_plateau":
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=max(1, patience // 2)
                )
            )

        # Train
        history = model.fit(
            X_train_mlp,
            y_train,
            validation_data=(X_valid_mlp, y_valid),
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=0,
        )

        # Measure RMSE on validation set
        preds = model.predict(X_valid_mlp, verbose=0)
        rmse = math.sqrt(mean_squared_error(y_valid, preds))
        return rmse

    return objective
