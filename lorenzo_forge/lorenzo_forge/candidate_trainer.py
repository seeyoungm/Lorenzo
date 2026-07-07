"""Builds a real tf.keras model from an ArchitectureSpec and trains it briefly
to get an empirical fitness score. This is the expensive "ground truth" step
that search.py calls many times to figure out which architecture is actually good."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec


def build_model(spec: ArchitectureSpec, profile: DataProfile) -> tf.keras.Model:
    if profile.task_type == "text":
        return _build_text_model(spec, profile)
    if profile.task_type == "timeseries":
        return _build_timeseries_model(spec, profile)

    inputs = layers.Input(shape=profile.input_shape)
    x = inputs

    if profile.task_type == "tabular":
        for _ in range(spec.num_blocks):
            x = layers.Dense(spec.units, activation=spec.activation)(x)
            if spec.dropout > 0:
                x = layers.Dropout(spec.dropout)(x)
    else:
        for _ in range(spec.num_blocks):
            x = layers.Conv2D(spec.units, spec.kernel_size, padding="same", activation=spec.activation)(x)
            if min(x.shape[1:3]) >= 2:
                x = layers.MaxPooling2D()(x)
            if spec.dropout > 0:
                x = layers.Dropout(spec.dropout)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(max(spec.units, 32), activation=spec.activation)(x)

    outputs = layers.Dense(profile.output_dim, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    optimizer = optimizers.Adam(spec.lr) if spec.optimizer == "adam" else optimizers.SGD(spec.lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _apply_encoder_stack(x, spec: ArchitectureSpec):
    """Shared recurrent/conv1d encoder stack for text and timeseries."""
    for i in range(spec.num_blocks):
        last = i == spec.num_blocks - 1
        if spec.encoder == "lstm":
            x = layers.LSTM(spec.units, return_sequences=not last)(x)
        elif spec.encoder == "gru":
            x = layers.GRU(spec.units, return_sequences=not last)(x)
        else:  # conv1d
            x = layers.Conv1D(spec.units, spec.kernel_size, padding="same", activation="relu")(x)
        if spec.dropout > 0:
            x = layers.Dropout(spec.dropout)(x)
    if spec.encoder == "conv1d":
        x = layers.GlobalMaxPooling1D()(x)
    return x


def _compile_classifier(inputs, x, spec: ArchitectureSpec, profile: DataProfile) -> tf.keras.Model:
    outputs = layers.Dense(profile.output_dim, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    optimizer = optimizers.Adam(spec.lr) if spec.optimizer == "adam" else optimizers.SGD(spec.lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _build_text_model(spec: ArchitectureSpec, profile: DataProfile) -> tf.keras.Model:
    seq_len = profile.input_shape[0]
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    x = layers.Embedding(input_dim=profile.vocab_size, output_dim=spec.embedding_dim)(inputs)
    x = _apply_encoder_stack(x, spec)
    return _compile_classifier(inputs, x, spec, profile)


def _build_timeseries_model(spec: ArchitectureSpec, profile: DataProfile) -> tf.keras.Model:
    # Real-valued sequence: (timesteps, channels) straight into the encoder,
    # no token embedding.
    inputs = layers.Input(shape=profile.input_shape)
    x = _apply_encoder_stack(inputs, spec)
    return _compile_classifier(inputs, x, spec, profile)


def evaluate_spec(
    spec: ArchitectureSpec,
    profile: DataProfile,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epochs: int = 4,
    batch_size: int = 32,
) -> float:
    X_train, y_train, X_val, y_val = data
    try:
        model = build_model(spec, profile)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        return float(history.history["val_accuracy"][-1])
    except Exception:
        return 0.0
