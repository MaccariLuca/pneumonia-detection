"""
Training callbacks module.
"""

import tf_keras as keras

from config import (
    MODEL_NAME,
    EARLY_STOP_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    LR_DECAY,
    LEARNING_RATE,
)


def get_callbacks() -> list:
    """
    Returns the list of Keras callbacks used during training.

    - ModelCheckpoint:        saves the best model by val_accuracy.
    - EarlyStopping:          stops training if val_accuracy doesn't improve.
    - ReduceLROnPlateau:      halves LR when val_loss stalls.
    - LearningRateScheduler: exponential decay every epoch.
    """
    return [
        keras.callbacks.ModelCheckpoint(
            MODEL_NAME,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (LR_DECAY ** epoch),
            verbose=0,
        ),
    ]
