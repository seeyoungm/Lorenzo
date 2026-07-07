import json

import numpy as np

from lorenzo_forge.dataset_builder import build_training_corpus
from lorenzo_forge.meta_model import train_scorer_model
from lorenzo_forge.release import make_splits, release
from lorenzo_forge.synthetic import generate_dataset, sample_tabular_profile


def _tiny_scorer():
    records = build_training_corpus(
        num_profiles=3, candidates_per_profile=3, search_epochs=1, seed=0, verbose=False, real_images=False
    )
    return train_scorer_model(records, epochs=3, verbose=0)


def test_make_splits_are_disjoint_and_cover():
    X = np.arange(200).reshape(200, 1).astype("float32")
    y = (np.arange(200) % 2).astype("int64")
    (Xtr, _), (Xval, _), (Xte, _) = make_splits(X, y, seed=1)
    ids = set(Xtr.ravel()) | set(Xval.ravel()) | set(Xte.ravel())
    assert len(ids) == 200  # partition covers everything, no overlap
    assert len(Xte) == 40 and len(Xval) == 40  # 20% / 20%


def test_release_writes_model_and_card(tmp_path):
    scorer = _tiny_scorer()
    rng = np.random.default_rng(0)
    profile = sample_tabular_profile(rng)
    Xtr, ytr, Xval, yval = generate_dataset(profile, rng)
    X = np.concatenate([Xtr, Xval])
    y = np.concatenate([ytr, yval])

    card = release(
        name="Test Tabular Alpha", scorer=scorer, X=X, y=y, task_type="tabular",
        out_dir=tmp_path, top_k=2, epochs=3, seed=0, verbose=False,
    )

    assert (tmp_path / "model.keras").exists()
    saved = json.loads((tmp_path / "model_card.json").read_text())
    assert saved["name"] == "Test Tabular Alpha"
    assert len(saved["candidates"]) == 2
    # winner must be the best test accuracy among the trained candidates
    best = max(c["test_accuracy"] for c in saved["candidates"])
    assert saved["winner"]["test_accuracy"] == best
    assert 0.0 <= saved["winner"]["test_accuracy"] <= 1.0
    assert saved["winner"]["num_params"] > 0
