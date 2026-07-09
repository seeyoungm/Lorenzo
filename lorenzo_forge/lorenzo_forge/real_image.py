"""Real image profiles for the corpus.

The synthetic image task (synthetic.py) turned out to be nearly
architecture-insensitive: a good CNN barely beats a weak one, so there is
little ranking signal to learn (see README v0.2 image results). Real datasets
fix this -- on actual MNIST / Fashion-MNIST digits and garments, a good conv
architecture measurably beats a bad one.

To give the meta-model PROFILE DIVERSITY from just two sources, each sampled
profile picks a dataset and then varies: which class subset, how many samples,
the resolution, and how much extra input noise. That spans a real range of
(input_size, output_dim, num_samples, difficulty)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from lorenzo_forge.profile import DataProfile

SOURCES = ("mnist", "fashion_mnist", "cifar10")
RESOLUTION_CHOICES = (32, 28, 20, 14)  # 32 = CIFAR-10 native; others MNIST-family sized
OUTPUT_DIM_RANGE = (2, 7)   # use 2..6 of the 10 available classes
NUM_SAMPLES_RANGE = (500, 1600)

_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load_source(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in _CACHE:
        loader = getattr(tf.keras.datasets, name)
        (x_train, y_train), (x_test, y_test) = loader.load_data()
        x = np.concatenate([x_train, x_test]).astype("float32") / 255.0
        y = np.concatenate([y_train, y_test]).ravel().astype("int64")
        if x.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
            x = x[..., None]
        _CACHE[name] = (x, y)
    return _CACHE[name]


def sample_real_image_profile(
    rng: np.random.Generator,
) -> tuple[DataProfile, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    source = str(rng.choice(SOURCES))
    x_all, y_all = _load_source(source)

    n_classes = int(rng.integers(*OUTPUT_DIM_RANGE))
    chosen = rng.choice(10, size=n_classes, replace=False)
    remap = {int(c): i for i, c in enumerate(chosen)}
    mask = np.isin(y_all, chosen)
    x_sub, y_sub = x_all[mask], y_all[mask]

    num_samples = int(min(rng.integers(*NUM_SAMPLES_RANGE), len(x_sub)))
    sel = rng.choice(len(x_sub), size=num_samples, replace=False)
    x_sub = x_sub[sel]
    y_sub = np.array([remap[int(v)] for v in y_sub[sel]], dtype="int64")

    resolution = int(rng.choice(RESOLUTION_CHOICES))
    if resolution != x_sub.shape[1]:
        x_sub = tf.image.resize(x_sub, (resolution, resolution)).numpy().astype("float32")

    noise_level = float(rng.uniform(0.0, 0.4))
    if noise_level > 0:
        x_sub = np.clip(x_sub + rng.normal(scale=noise_level, size=x_sub.shape).astype("float32"), 0.0, 1.0)

    classes, counts = np.unique(y_sub, return_counts=True)
    probs = counts / counts.sum()
    k = len(classes)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0

    profile = DataProfile(
        task_type="image",
        input_shape=(resolution, resolution, x_sub.shape[-1]),
        output_dim=k,
        num_samples=len(x_sub),
        class_balance_entropy=entropy,
        noise_level=noise_level,
    )

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_sub, y_sub, test_size=0.25, random_state=int(rng.integers(0, 2**31 - 1)), stratify=y_sub
    )
    return profile, (x_tr, y_tr, x_val, y_val)
