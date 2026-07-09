import numpy as np

from lorenzo_forge.candidate_trainer import build_model, evaluate_spec
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec
from lorenzo_forge.synthetic import generate_dataset


def test_build_model_tabular_output_shape():
    profile = DataProfile("tabular", (10,), 3, 200, 1.0, 0.2)
    spec = ArchitectureSpec("tabular", num_blocks=2, units=16, activation="relu", dropout=0.0, optimizer="adam", lr=1e-3)
    model = build_model(spec, profile)
    assert model.output_shape == (None, 3)


def test_build_model_image_output_shape():
    profile = DataProfile("image", (16, 16, 1), 4, 200, 1.0, 0.2)
    spec = ArchitectureSpec("image", num_blocks=2, units=8, activation="relu", dropout=0.0, optimizer="adam", lr=1e-3, kernel_size=3)
    model = build_model(spec, profile)
    assert model.output_shape == (None, 4)


def test_build_model_image_residual_block_output_shape():
    profile = DataProfile("image", (16, 16, 1), 4, 200, 1.0, 0.2)
    spec = ArchitectureSpec(
        "image", num_blocks=2, units=8, activation="relu", dropout=0.0, optimizer="adam", lr=1e-3,
        kernel_size=3, block_style="residual",
    )
    model = build_model(spec, profile)
    assert model.output_shape == (None, 4)


def test_build_model_image_gap_pool_style_output_shape():
    profile = DataProfile("image", (16, 16, 1), 4, 200, 1.0, 0.2)
    spec = ArchitectureSpec(
        "image", num_blocks=2, units=8, activation="relu", dropout=0.0, optimizer="adam", lr=1e-3,
        kernel_size=3, pool_style="gap",
    )
    model = build_model(spec, profile)
    assert model.output_shape == (None, 4)


def test_build_model_tabular_residual_block_output_shape():
    profile = DataProfile("tabular", (10,), 3, 200, 1.0, 0.2)
    spec = ArchitectureSpec(
        "tabular", num_blocks=2, units=16, activation="relu", dropout=0.0, optimizer="adam", lr=1e-3,
        block_style="residual",
    )
    model = build_model(spec, profile)
    assert model.output_shape == (None, 3)


def test_evaluate_spec_returns_accuracy_in_range():
    rng = np.random.default_rng(7)
    profile = DataProfile("tabular", (10,), 2, 200, 1.0, 0.1)
    data = generate_dataset(profile, rng)
    spec = ArchitectureSpec("tabular", num_blocks=1, units=16, activation="relu", dropout=0.0, optimizer="adam", lr=1e-2)
    score = evaluate_spec(spec, profile, data, epochs=1)
    assert 0.0 <= score <= 1.0
