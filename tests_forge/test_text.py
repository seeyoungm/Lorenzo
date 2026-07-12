import numpy as np

from lorenzo_forge.candidate_trainer import build_model
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import (
    ENCODER_CHOICES,
    ArchitectureSpec,
    decode_from_indices,
    enumerate_specs,
    sample_random_spec,
)


def test_text_spec_roundtrip_all_encoders():
    for enc in ENCODER_CHOICES:
        kernel = 3 if enc == "conv1d" else None
        spec = ArchitectureSpec(
            "text", num_blocks=2, units=64, activation="tanh", dropout=0.2,
            optimizer="adam", lr=1e-3, embedding_dim=32, encoder=enc, kernel_size=kernel,
        )
        decoded = decode_from_indices("text", spec.label_indices())
        assert decoded == spec


def test_enumerate_text_covers_encoders_and_is_bounded():
    specs = list(enumerate_specs("text"))
    assert len(specs) == 36864  # 41472 minus 4608 deep-recurrent(3+ blocks) lr=1e-2 combos excluded
    assert {s.encoder for s in specs} == set(ENCODER_CHOICES)
    # conv1d specs carry a kernel; recurrent specs do not
    assert all(s.kernel_size is not None for s in specs if s.encoder == "conv1d")
    assert all(s.kernel_size is None for s in specs if s.encoder in ("lstm", "gru"))


def test_build_text_models_have_correct_output_shape():
    profile = DataProfile("text", (50,), 3, 200, 1.0, 0.0, vocab_size=500)
    for enc in ENCODER_CHOICES:
        kernel = 3 if enc == "conv1d" else None
        spec = ArchitectureSpec(
            "text", 1, 16, "tanh", 0.0, "adam", 1e-3, embedding_dim=16, encoder=enc, kernel_size=kernel
        )
        model = build_model(spec, profile)
        assert model.output_shape == (None, 3)


def test_sample_text_spec_is_valid():
    rng = np.random.default_rng(0)
    spec = sample_random_spec("text", rng)
    assert spec.encoder in ENCODER_CHOICES
    assert spec.embedding_dim in (16, 32, 64)
    if spec.encoder != "conv1d":
        assert spec.kernel_size is None
