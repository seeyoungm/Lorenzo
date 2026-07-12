import numpy as np

from lorenzo_forge.search_space import (
    HEADS,
    ArchitectureSpec,
    decode_from_indices,
    enumerate_specs,
    is_valid_spec,
    sample_random_spec,
)


def test_sample_random_spec_tabular_has_no_kernel():
    rng = np.random.default_rng(0)
    spec = sample_random_spec("tabular", rng)
    assert spec.kernel_size is None


def test_sample_random_spec_image_has_kernel():
    rng = np.random.default_rng(0)
    spec = sample_random_spec("image", rng)
    assert spec.kernel_size in (3, 5)


def test_label_indices_roundtrip():
    rng = np.random.default_rng(3)
    for task_type in ("tabular", "image"):
        spec = sample_random_spec(task_type, rng)
        indices = spec.label_indices()
        assert set(indices.keys()) == set(HEADS.keys())
        decoded = decode_from_indices(task_type, indices)
        assert decoded == spec


def test_to_dict_block_count_matches_num_blocks():
    rng = np.random.default_rng(5)
    spec = sample_random_spec("image", rng)
    assert len(spec.to_dict()["blocks"]) == spec.num_blocks


def test_is_valid_spec_excludes_deep_recurrent_high_lr():
    bad = ArchitectureSpec(
        "text", num_blocks=3, units=128, activation="tanh", dropout=0.0,
        optimizer="adam", lr=1e-2, embedding_dim=32, encoder="bigru",
    )
    assert not is_valid_spec(bad)
    # same stack at a safe lr, and a shallow stack at high lr, are both fine
    assert is_valid_spec(ArchitectureSpec(
        "text", num_blocks=3, units=128, activation="tanh", dropout=0.0,
        optimizer="adam", lr=1e-3, embedding_dim=32, encoder="bigru",
    ))
    assert is_valid_spec(ArchitectureSpec(
        "text", num_blocks=2, units=128, activation="tanh", dropout=0.0,
        optimizer="adam", lr=1e-2, embedding_dim=32, encoder="bigru",
    ))


def test_is_valid_spec_caps_wide_image_residual():
    bad = ArchitectureSpec(
        "image", num_blocks=2, units=256, activation="relu", dropout=0.0,
        optimizer="adam", lr=1e-3, kernel_size=3, block_style="residual",
    )
    assert not is_valid_spec(bad)
    # 128-unit residual (the CIFAR winner width) and 256-unit plain are fine
    assert is_valid_spec(ArchitectureSpec(
        "image", num_blocks=2, units=128, activation="relu", dropout=0.0,
        optimizer="adam", lr=1e-3, kernel_size=3, block_style="residual",
    ))
    assert is_valid_spec(ArchitectureSpec(
        "image", num_blocks=2, units=256, activation="relu", dropout=0.0,
        optimizer="adam", lr=1e-3, kernel_size=3, block_style="plain",
    ))


def test_sampler_and_enumerator_agree_on_validity():
    rng = np.random.default_rng(11)
    for task_type in ("tabular", "image", "text", "timeseries"):
        for _ in range(50):
            assert is_valid_spec(sample_random_spec(task_type, rng))
        assert all(is_valid_spec(s) for s in enumerate_specs(task_type))
