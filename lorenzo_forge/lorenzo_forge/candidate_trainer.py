"""Builds a real tf.keras model from an ArchitectureSpec and trains it briefly
to get an empirical fitness score. This is the expensive "ground truth" step
that search.py calls many times to figure out which architecture is actually good."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec


def _residual_dense_block(x, units: int, activation: str):
    """Two BN'd Dense layers + skip connection; shortcut gets a linear
    projection only when the incoming width differs from `units` (i.e. the
    very first block of a stack, since every block after that already has
    width `units`)."""
    shortcut = x
    y = layers.Dense(units, use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    y = layers.Dense(units, use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y])
    return layers.Activation(activation)(y)


def _residual_conv_block(x, units: int, kernel_size: int, activation: str):
    """Two BN'd Conv2D layers + skip connection (1x1 projection on channel
    mismatch). Both convs use padding="same"/stride 1 so spatial dims match
    the shortcut -- any pooling happens once, after the whole block."""
    shortcut = x
    y = layers.Conv2D(units, kernel_size, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    y = layers.Conv2D(units, kernel_size, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != units:
        shortcut = layers.Conv2D(units, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y])
    return layers.Activation(activation)(y)


def _residual_conv1d_block(x, units: int, kernel_size: int):
    """Conv1D analogue of `_residual_conv_block`, used by the conv1d text /
    timeseries encoder. Fixed relu activation, matching the plain conv1d
    branch's existing (non-configurable) activation."""
    shortcut = x
    y = layers.Conv1D(units, kernel_size, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(units, kernel_size, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != units:
        shortcut = layers.Conv1D(units, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y])
    return layers.Activation("relu")(y)


def build_model(spec: ArchitectureSpec, profile: DataProfile) -> tf.keras.Model:
    if profile.task_type == "text":
        return _build_text_model(spec, profile)
    if profile.task_type == "timeseries":
        return _build_timeseries_model(spec, profile)

    inputs = layers.Input(shape=profile.input_shape)
    x = inputs

    if profile.task_type == "tabular":
        for _ in range(spec.num_blocks):
            if spec.block_style == "residual":
                x = _residual_dense_block(x, spec.units, spec.activation)
            else:
                x = layers.Dense(spec.units, use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation(spec.activation)(x)
            if spec.dropout > 0:
                x = layers.Dropout(spec.dropout)(x)
    else:
        for _ in range(spec.num_blocks):
            if spec.block_style == "residual":
                x = _residual_conv_block(x, spec.units, spec.kernel_size, spec.activation)
            else:
                x = layers.Conv2D(spec.units, spec.kernel_size, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation(spec.activation)(x)
            if min(x.shape[1:3]) >= 2:
                x = layers.MaxPooling2D()(x)
            if spec.dropout > 0:
                x = layers.Dropout(spec.dropout)(x)
        if spec.pool_style == "gap":
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.Flatten()(x)
            x = layers.Dense(max(spec.units, 32), activation=spec.activation)(x)

    outputs = layers.Dense(profile.output_dim, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    optimizer = optimizers.Adam(spec.lr) if spec.optimizer == "adam" else optimizers.SGD(spec.lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _recurrent_layer(kind: str, units: int, return_sequences: bool):
    rnn = layers.LSTM if kind in ("lstm", "bilstm") else layers.GRU
    layer = rnn(units, return_sequences=return_sequences)
    if kind in ("bilstm", "bigru"):
        return layers.Bidirectional(layer)
    return layer


def _apply_encoder_stack(x, spec: ArchitectureSpec):
    """Shared recurrent/conv1d encoder stack for text and timeseries.
    Recurrent encoders may be uni- (lstm/gru) or bidirectional (bilstm/bigru)
    and are left as-is (no BatchNorm/residual -- non-standard across
    recurrent steps, out of scope). The conv1d encoder additionally supports
    BatchNorm + residual blocks and a GAP1D vs GlobalMaxPooling1D choice via
    block_style/pool_style."""
    for i in range(spec.num_blocks):
        last = i == spec.num_blocks - 1
        if spec.encoder == "conv1d":
            if spec.block_style == "residual":
                x = _residual_conv1d_block(x, spec.units, spec.kernel_size)
            else:
                x = layers.Conv1D(spec.units, spec.kernel_size, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
        else:
            x = _recurrent_layer(spec.encoder, spec.units, return_sequences=not last)(x)
        if spec.dropout > 0:
            x = layers.Dropout(spec.dropout)(x)
    if spec.encoder == "conv1d":
        if spec.pool_style == "gap":
            x = layers.GlobalAveragePooling1D()(x)
        else:
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
    finally:
        # Corpus building calls this hundreds of times per run (num_profiles x
        # num_candidates); without clearing the session, Keras' global layer
        # name/uid registries and tf.function retracing caches grow
        # unbounded across models and the process's memory climbs into the
        # tens of GB well before finishing.
        tf.keras.backend.clear_session()
