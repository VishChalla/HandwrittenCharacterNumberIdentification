"""The CNN architecture."""

from __future__ import annotations

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.optimizers import Adam


def build_model(augment: bool = True) -> Sequential:
    """Two convolution blocks and a dense head.

    Augmentation layers sit inside the model, so they are active only during
    training and the saved model needs no separate preprocessing code.
    """
    layers = [Input(shape=(28, 28, 1))]
    if augment:
        layers += [RandomRotation(10 / 360), RandomZoom(0.1), RandomTranslation(0.1, 0.1)]
    layers += [
        Conv2D(32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
