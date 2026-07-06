import numpy as np

from lorenzo_forge.real_image import sample_real_image_profile


def test_real_image_profile_shapes_and_labels():
    rng = np.random.default_rng(0)
    profile, (x_tr, y_tr, x_val, y_val) = sample_real_image_profile(rng)

    assert profile.task_type == "image"
    assert len(profile.input_shape) == 3
    # image data matches the declared profile shape
    assert x_tr.shape[1:] == profile.input_shape
    assert x_val.shape[1:] == profile.input_shape
    # labels are remapped to a contiguous 0..k-1 range
    labels = set(np.unique(y_tr)) | set(np.unique(y_val))
    assert labels == set(range(profile.output_dim))
    # pixels normalized to [0, 1]
    assert x_tr.min() >= 0.0 and x_tr.max() <= 1.0


def test_real_image_profile_varies_across_samples():
    rng = np.random.default_rng(1)
    shapes = {sample_real_image_profile(rng)[0].input_shape for _ in range(6)}
    dims = {sample_real_image_profile(np.random.default_rng(s))[0].output_dim for s in range(6)}
    # resolution and/or class-count variety is present across draws
    assert len(shapes) > 1 or len(dims) > 1
