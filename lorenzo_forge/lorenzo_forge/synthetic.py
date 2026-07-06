"""Synthetic profile + dataset generation used to bootstrap training labels
for the meta-model (there's no public "which architecture wins" dataset, so
we manufacture (profile, empirically-best-architecture) pairs ourselves)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lorenzo_forge.profile import DataProfile

TABULAR_FEATURE_RANGE = (8, 40)
IMAGE_SIDE_CHOICES = [16, 24, 32]
IMAGE_CHANNEL_CHOICES = [1, 3]
OUTPUT_DIM_RANGE = (2, 6)
NUM_SAMPLES_RANGE = (400, 1600)


def sample_random_profile(rng: np.random.Generator) -> DataProfile:
    task_type = str(rng.choice(["tabular", "image"]))
    output_dim = int(rng.integers(*OUTPUT_DIM_RANGE))
    num_samples = int(rng.integers(*NUM_SAMPLES_RANGE))
    noise_level = float(rng.uniform(0.05, 0.5))
    class_balance_entropy = float(rng.uniform(0.6, 1.0))

    if task_type == "tabular":
        n_features = int(rng.integers(*TABULAR_FEATURE_RANGE))
        input_shape = (n_features,)
    else:
        side = int(rng.choice(IMAGE_SIDE_CHOICES))
        channels = int(rng.choice(IMAGE_CHANNEL_CHOICES))
        input_shape = (side, side, channels)

    return DataProfile(
        task_type=task_type,
        input_shape=input_shape,
        output_dim=output_dim,
        num_samples=num_samples,
        class_balance_entropy=class_balance_entropy,
        noise_level=noise_level,
    )


def generate_dataset(profile: DataProfile, rng: np.random.Generator):
    """Returns (X_train, y_train, X_val, y_val) matching the given profile."""
    if profile.task_type == "tabular":
        n_features = profile.input_shape[0]
        n_informative = max(2, min(n_features, int(n_features * (1 - profile.noise_level))))
        weights = _entropy_to_class_weights(profile.class_balance_entropy, profile.output_dim, rng)
        X, y = make_classification(
            n_samples=profile.num_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=profile.output_dim,
            n_clusters_per_class=1,
            weights=weights,
            flip_y=profile.noise_level * 0.3,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        X = X.astype("float32")
    else:
        h, w, c = profile.input_shape
        base_patterns = rng.normal(size=(profile.output_dim, h, w, c)).astype("float32")
        weights = _entropy_to_class_weights(profile.class_balance_entropy, profile.output_dim, rng)
        y = rng.choice(profile.output_dim, size=profile.num_samples, p=weights).astype("int64")
        noise = rng.normal(scale=profile.noise_level, size=(profile.num_samples, h, w, c)).astype("float32")
        X = base_patterns[y] + noise
        X = 1.0 / (1.0 + np.exp(-X))  # squash to a [0, 1]-ish range like normalized pixels

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=int(rng.integers(0, 2**31 - 1)), stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, y_train, X_val, y_val


def _entropy_to_class_weights(entropy: float, k: int, rng: np.random.Generator) -> list[float]:
    """Maps a target balance entropy (1=uniform, lower=skewed) to a concrete class weight vector."""
    if k <= 1:
        return [1.0]
    skew = max(0.0, 1.0 - entropy)
    raw = rng.dirichlet(alpha=[max(0.5, 5.0 * (1 - skew))] * k)
    uniform = np.full(k, 1.0 / k)
    weights = (1 - skew) * uniform + skew * raw
    weights = weights / weights.sum()
    return weights.tolist()
