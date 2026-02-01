"""
Model definition module.
Contains the data augmentation pipeline and the CNN architecture.
"""

import tf_keras as keras
from tf_keras import layers

from config import (
    IMG_SIZE,
    DROPOUT_RATES,
    AUG_ROTATION,
    AUG_ZOOM,
    AUG_TRANSLATION,
    LEARNING_RATE,
)


def build_augmentation() -> keras.Sequential:
    """
    Returns a Sequential model that applies random augmentations.
    Only active during training — automatically disabled during inference.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUG_ROTATION),
        layers.RandomZoom(AUG_ZOOM),
        layers.RandomTranslation(AUG_TRANSLATION, AUG_TRANSLATION),
    ], name="augmentation")


def build_model() -> keras.Model:
    """
    Builds and compiles the CNN.

    Architecture:
        - Augmentation layer
        - 4 convolutional blocks (16 → 32 → 64 → 128 filters)
          each with BatchNorm + ReLU + MaxPool + Dropout
        - 2 dense layers (256 → 128) with BatchNorm + Dropout
        - Softmax output (2 classes)

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Augmentation
        build_augmentation(),

        # Block 1 — 16 filters
        layers.Conv2D(16, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATES[0]),

        # Block 2 — 32 filters
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATES[1]),

        # Block 3 — 64 filters
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATES[2]),

        # Block 4 — 128 filters
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATES[3]),

        # Dense 1
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(DROPOUT_RATES[4]),

        # Dense 2
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(DROPOUT_RATES[5]),

        # Output
        layers.Dense(2, activation="softmax",
                     kernel_initializer=keras.initializers.GlorotUniform(seed=42)),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
