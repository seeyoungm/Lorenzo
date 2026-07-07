import numpy as np

from lorenzo_forge.candidate_trainer import build_model
from lorenzo_forge.datasets import build_timeseries_v1
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.release import BUILTIN_TIMESERIES_DOMAINS, load_domain
from lorenzo_forge.search_space import (
    ENCODER_CHOICES,
    ArchitectureSpec,
    decode_from_indices,
    enumerate_specs,
    sample_random_spec,
)
from lorenzo_forge.timeseries_data import sample_timeseries_profile


def test_timeseries_spec_has_no_embedding_but_has_encoder():
    spec = sample_random_spec("timeseries", np.random.default_rng(0))
    assert spec.encoder in ENCODER_CHOICES
    assert spec.embedding_dim is None
    assert decode_from_indices("timeseries", spec.label_indices()) == spec


def test_enumerate_timeseries_size():
    specs = list(enumerate_specs("timeseries"))
    assert len(specs) == 3456
    assert all(s.embedding_dim is None for s in specs)


def test_build_timeseries_models_output_shape():
    profile = DataProfile("timeseries", (40, 2), 3, 200, 1.0, 0.5)
    for enc in ENCODER_CHOICES:
        kernel = 3 if enc == "conv1d" else None
        spec = ArchitectureSpec("timeseries", 1, 16, "tanh", 0.0, "adam", 1e-3, kernel_size=kernel, encoder=enc)
        assert build_model(spec, profile).output_shape == (None, 3)


def test_sample_timeseries_profile_shapes():
    profile, (xtr, ytr, xv, yv) = sample_timeseries_profile(np.random.default_rng(1))
    assert profile.task_type == "timeseries"
    assert len(profile.input_shape) == 2  # (timesteps, channels)
    assert xtr.shape[1:] == profile.input_shape


def test_timeseries_v1_deterministic_and_domain():
    X1, y1 = build_timeseries_v1()
    X2, y2 = build_timeseries_v1()
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert "timeseries_v1" in BUILTIN_TIMESERIES_DOMAINS
    X, y, task_type, vocab = load_domain("timeseries_v1")
    assert task_type == "timeseries" and vocab == 0 and X.ndim == 3
