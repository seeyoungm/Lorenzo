"""Synthetic time-series (sequence) classification profiles — fourth domain.

Unlike static images, synthetic time-series carries genuine architecture signal:
each class is defined by (a) a global dominant frequency and (b) a short local
motif stamped at a random position. Detecting the position-invariant motif needs
conv1d, and the frequency component rewards recurrent/deeper models, so a good
sequence architecture clearly beats a weak one. Profiles vary by length,
channels, class count, noise, and sample count for diversity."""

from __future__ import annotations

import numpy as np

from lorenzo_forge.profile import DataProfile

TIMESTEPS_CHOICES = (40, 60, 80)
CHANNELS_CHOICES = (1, 2, 3)
OUTPUT_DIM_RANGE = (2, 6)
NUM_SAMPLES_RANGE = (600, 1500)


def generate_timeseries(
    n_samples: int, timesteps: int, channels: int, n_classes: int, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    motif_len = max(3, timesteps // 4)
    motifs = rng.normal(size=(n_classes, motif_len, channels)).astype("float32")
    freqs = rng.uniform(0.5, 4.0, size=n_classes).astype("float32")
    phases = rng.uniform(0, 2 * np.pi, size=n_classes).astype("float32")
    t = np.linspace(0, 2 * np.pi, timesteps, dtype="float32")

    y = rng.integers(0, n_classes, size=n_samples).astype("int64")
    X = rng.normal(scale=noise, size=(n_samples, timesteps, channels)).astype("float32")
    for i in range(n_samples):
        c = y[i]
        X[i] += 0.6 * np.sin(freqs[c] * t + phases[c])[:, None]  # global class frequency
        pos = int(rng.integers(0, timesteps - motif_len + 1))    # local motif, random position
        X[i, pos:pos + motif_len, :] += motifs[c]
    return X, y


def sample_timeseries_profile(
    rng: np.random.Generator,
) -> tuple[DataProfile, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    timesteps = int(rng.choice(TIMESTEPS_CHOICES))
    channels = int(rng.choice(CHANNELS_CHOICES))
    n_classes = int(rng.integers(*OUTPUT_DIM_RANGE))
    num_samples = int(rng.integers(*NUM_SAMPLES_RANGE))
    noise = float(rng.uniform(0.3, 1.0))

    X, y = generate_timeseries(num_samples, timesteps, channels, n_classes, noise, rng)

    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    k = len(classes)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0

    profile = DataProfile(
        task_type="timeseries",
        input_shape=(timesteps, channels),
        output_dim=k,
        num_samples=num_samples,
        class_balance_entropy=entropy,
        noise_level=noise,
    )

    split = int(num_samples * 0.75)
    perm = rng.permutation(num_samples)
    tr, va = perm[:split], perm[split:]
    return profile, (X[tr], y[tr], X[va], y[va])
