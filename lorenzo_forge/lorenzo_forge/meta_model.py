"""The meta-model: a small multi-head neural network trained on
(data-profile-features -> empirically-best-architecture) pairs produced by
dataset_builder.py. This is the actual "learning" component of Lorenzo Forge
-- everything upstream (search.py, candidate_trainer.py) exists only to
generate its training labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from lorenzo_forge.profile import DataProfile, FEATURE_NAMES
from lorenzo_forge.search_space import ArchitectureSpec, HEADS, decode_from_indices


def build_meta_model(input_dim: int = len(FEATURE_NAMES), hidden_units: int = 64) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="profile_features")
    x = layers.Dense(hidden_units, activation="relu")(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)

    outputs = {name: layers.Dense(size, activation="softmax", name=name)(x) for name, size in HEADS.items()}
    model = models.Model(inputs, outputs, name="lorenzo_forge_meta_model")
    model.compile(
        optimizer="adam",
        loss={name: "sparse_categorical_crossentropy" for name in HEADS},
        metrics={name: "accuracy" for name in HEADS},
    )
    return model


def _corpus_to_arrays(records: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    X = np.stack([DataProfile.from_dict(r["profile"]).to_feature_vector() for r in records])
    Y: dict[str, list[int]] = {name: [] for name in HEADS}
    for r in records:
        spec = ArchitectureSpec.from_raw_dict(r["best_spec"])
        indices = spec.label_indices()
        for name in HEADS:
            Y[name].append(indices[name])
    Y_arrays = {name: np.array(vals, dtype="int64") for name, vals in Y.items()}
    return X, Y_arrays


def train_meta_model(records: list[dict], epochs: int = 30, batch_size: int = 16, verbose: int = 0) -> tf.keras.Model:
    X, Y = _corpus_to_arrays(records)
    model = build_meta_model(input_dim=X.shape[1])
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def save_meta_model(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_meta_model(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def recommend(model: tf.keras.Model, profile: DataProfile) -> tuple[ArchitectureSpec, dict[str, float]]:
    x = profile.to_feature_vector()[None, :]
    preds = model.predict(x, verbose=0)
    # Keras returns a list in output-layer order when the model has multiple named outputs.
    pred_by_name = dict(zip(HEADS.keys(), preds)) if isinstance(preds, list) else preds
    indices = {name: int(np.argmax(pred_by_name[name][0])) for name in HEADS}
    confidences = {name: float(np.max(pred_by_name[name][0])) for name in HEADS}
    spec = decode_from_indices(profile.task_type, indices)
    return spec, confidences
