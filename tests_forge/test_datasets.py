import numpy as np

from lorenzo_forge.datasets import TABULAR_V1_SPEC, build_tabular_v1
from lorenzo_forge.release import BUILTIN_TABULAR_DOMAINS, load_domain


def test_tabular_v1_is_deterministic():
    X1, y1 = build_tabular_v1()
    X2, y2 = build_tabular_v1()
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_tabular_v1_matches_spec():
    X, y = build_tabular_v1()
    assert X.shape == (TABULAR_V1_SPEC["n_samples"], TABULAR_V1_SPEC["n_features"])
    assert set(np.unique(y)) == set(range(TABULAR_V1_SPEC["n_classes"]))
    assert X.dtype == np.float32 and y.dtype == np.int64


def test_load_domain_tabular_v1():
    assert "tabular_v1" in BUILTIN_TABULAR_DOMAINS
    X, y, task_type, vocab = load_domain("tabular_v1")
    assert task_type == "tabular" and vocab == 0
    assert X.shape[1] == 30
