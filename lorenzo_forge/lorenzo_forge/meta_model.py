"""The meta-model (v0.2): a scorer/ranking network.

Instead of classifying the single "winning" architecture per profile (v0.1,
which starved on ~1 noisy label per profile), the scorer learns a regression

    (profile features  +  architecture one-hot)  ->  expected val accuracy

from EVERY candidate trial in the corpus (N labels per profile, and ties are
no longer a problem -- near-equal architectures just get near-equal targets).

At recommend time we enumerate the whole search space, score each architecture,
and return the highest-scoring one. This is a tiny weight-free NAS predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from lorenzo_forge.profile import DataProfile, FEATURE_NAMES
from lorenzo_forge.search_space import ARCH_FEATURE_DIM, ArchitectureSpec, enumerate_specs

SCORER_INPUT_DIM = len(FEATURE_NAMES) + ARCH_FEATURE_DIM


def build_scorer_model(input_dim: int = SCORER_INPUT_DIM, hidden_units: int = 96) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="profile_and_arch")
    x = layers.Dense(hidden_units, activation="relu")(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(hidden_units, activation="relu")(x)
    score = layers.Dense(1, activation="sigmoid", name="score")(x)  # target is accuracy in [0, 1]
    model = models.Model(inputs, score, name="lorenzo_forge_scorer")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _encode(profile: DataProfile, spec: ArchitectureSpec) -> np.ndarray:
    return np.concatenate([profile.to_feature_vector(), spec.to_feature_vector()])


def corpus_to_scorer_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[float] = []
    for r in records:
        profile = DataProfile.from_dict(r["profile"])
        for trial in r["trials"]:
            spec = ArchitectureSpec.from_raw_dict(trial["spec"])
            X.append(_encode(profile, spec))
            y.append(float(trial["score"]))
    return np.stack(X).astype("float32"), np.array(y, dtype="float32")


def train_scorer_model(records: list[dict], epochs: int = 120, batch_size: int = 64, verbose: int = 0) -> tf.keras.Model:
    X, y = corpus_to_scorer_arrays(records)
    model = build_scorer_model(input_dim=X.shape[1])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def save_meta_model(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_meta_model(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def score_specs(model: tf.keras.Model, profile: DataProfile, specs: list[ArchitectureSpec]) -> np.ndarray:
    feats = np.stack([_encode(profile, s) for s in specs]).astype("float32")
    return model.predict(feats, verbose=0).ravel()


def recommend(
    model: tf.keras.Model, profile: DataProfile, top_k: int = 1, tie_tolerance: float = 0.02
) -> tuple[ArchitectureSpec, float, list[tuple[ArchitectureSpec, float]]]:
    """Enumerate the search space, score every architecture, return the best.
    Returns (best_spec, predicted_accuracy, top_k ranked [(spec, pred), ...]).

    The scorer barely separates its top candidates -- across every domain the
    top-k predictions span <0.025, i.e. within its own noise. Ranking those by
    raw score alone lets a fixed top-k budget get spent fully-training expensive
    residual/wide candidates that predict a noise-level fraction higher than a
    cheap equivalent and never actually win (documented repeatedly: image
    residuals at ~50min/candidate never took a release). So we bucket scores
    that fall within `tie_tolerance` and, within a bucket, prefer the cheaper
    architecture (same complexity_proxy tie-break search.py already uses when
    recording corpus labels). This trims release compute 20-48% per domain while
    the top-1 predicted accuracy drops at most `tie_tolerance`."""
    specs = list(enumerate_specs(profile.task_type))
    scores = score_specs(model, profile, specs)

    def rank_key(i: int) -> tuple:
        bucket = round(float(scores[i]) / tie_tolerance) if tie_tolerance > 0 else float(scores[i])
        return (-bucket, specs[i].complexity_proxy(), specs[i].describe())

    order = sorted(range(len(specs)), key=rank_key)
    ranked = [(specs[i], float(scores[i])) for i in order[:top_k]]
    best_spec, best_score = ranked[0]
    return best_spec, best_score, ranked
