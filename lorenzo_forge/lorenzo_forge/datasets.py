"""Reproducible, versioned built-in datasets for releases.

Unlike the corpus's random synthetic profiles, a *release* must be reproducible:
the exact same data every time. `tabular_v1` is a frozen tabular benchmark
(fixed spec + seed) so `Lorenzo Tabular 1.0` can be regenerated bit-for-bit
instead of depending on a throwaway .npz."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

# Frozen spec for the tabular release task. Do not change these numbers once a
# release ships against them; bump to tabular_v2 instead.
TABULAR_V1_SPEC = {
    "n_samples": 8000,
    "n_features": 30,
    "n_informative": 18,
    "n_redundant": 6,
    "n_classes": 4,
    "n_clusters_per_class": 2,
    "flip_y": 0.03,
    "weights": [0.30, 0.27, 0.23, 0.20],  # mild, fixed class imbalance
    "seed": 20240501,
}


def build_tabular_v1() -> tuple[np.ndarray, np.ndarray]:
    """Deterministically build the frozen tabular_v1 dataset. Returns (X, y)."""
    s = TABULAR_V1_SPEC
    X, y = make_classification(
        n_samples=s["n_samples"],
        n_features=s["n_features"],
        n_informative=s["n_informative"],
        n_redundant=s["n_redundant"],
        n_classes=s["n_classes"],
        n_clusters_per_class=s["n_clusters_per_class"],
        weights=s["weights"],
        flip_y=s["flip_y"],
        random_state=s["seed"],
    )
    return X.astype("float32"), y.astype("int64")


# Frozen spec for the time-series release task. Noise tuned (probe) so a good
# conv1d lands well below saturation while weak archs fail — a non-trivial task.
TIMESERIES_V1_SPEC = {
    "n_samples": 6000,
    "timesteps": 64,
    "channels": 3,
    "n_classes": 5,
    "noise": 1.8,
    "seed": 20240501,
}


def build_timeseries_v1() -> tuple[np.ndarray, np.ndarray]:
    """Deterministically build the frozen timeseries_v1 dataset. Returns (X, y)."""
    from lorenzo_forge.timeseries_data import generate_timeseries

    s = TIMESERIES_V1_SPEC
    rng = np.random.default_rng(s["seed"])
    return generate_timeseries(
        s["n_samples"], s["timesteps"], s["channels"], s["n_classes"], s["noise"], rng
    )
