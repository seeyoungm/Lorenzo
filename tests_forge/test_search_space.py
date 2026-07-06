import numpy as np

from lorenzo_forge.search_space import HEADS, decode_from_indices, sample_random_spec


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
