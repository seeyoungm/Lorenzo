import numpy as np

from lorenzo_forge.profile import DataProfile, FEATURE_NAMES


def test_feature_vector_length_and_flags():
    profile = DataProfile(
        task_type="image",
        input_shape=(16, 16, 3),
        output_dim=4,
        num_samples=500,
        class_balance_entropy=0.9,
        noise_level=0.1,
    )
    vec = profile.to_feature_vector()
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec[0] == 0.0  # is_tabular
    assert vec[1] == 1.0  # is_image


def test_input_size_flattens_shape():
    profile = DataProfile("image", (4, 4, 3), 2, 100, 1.0, 0.1)
    assert profile.input_size == 48


def test_from_arrays_computes_balance_entropy_for_balanced_labels():
    X = np.zeros((100, 10))
    y = np.array([0] * 50 + [1] * 50)
    profile = DataProfile.from_arrays(X, y, task_type="tabular")
    assert profile.output_dim == 2
    assert profile.class_balance_entropy > 0.99


def test_dict_roundtrip():
    profile = DataProfile("tabular", (10,), 3, 200, 0.8, 0.3)
    restored = DataProfile.from_dict(profile.to_dict())
    assert restored == profile
