from lorenzo_forge.dataset_builder import build_training_corpus
from lorenzo_forge.meta_model import SCORER_INPUT_DIM, build_scorer_model, recommend, train_scorer_model
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec, enumerate_specs


def test_scorer_model_output_is_scalar():
    model = build_scorer_model()
    assert model.output_shape == (None, 1)
    assert model.input_shape == (None, SCORER_INPUT_DIM)


def test_enumerate_specs_sizes():
    assert len(list(enumerate_specs("tabular"))) == 4 * 4 * 1 * 2 * 3 * 2 * 3
    assert len(list(enumerate_specs("image"))) == 4 * 4 * 2 * 2 * 3 * 2 * 3


def test_corpus_records_include_all_trials():
    records = build_training_corpus(
        num_profiles=2, candidates_per_profile=3, search_epochs=1, seed=2, verbose=False, real_images=False, domains=("tabular",)
    )
    for r in records:
        assert len(r["trials"]) >= 1
        for t in r["trials"]:
            ArchitectureSpec.from_raw_dict(t["spec"])
            assert 0.0 <= t["score"] <= 1.0


def test_scorer_end_to_end_recommend():
    records = build_training_corpus(
        num_profiles=3, candidates_per_profile=3, search_epochs=1, seed=1, verbose=False, real_images=False, domains=("tabular",)
    )
    model = train_scorer_model(records, epochs=3, verbose=0)

    profile = DataProfile("tabular", (10,), 3, 500, 1.0, 0.2)
    spec, predicted_acc, ranked = recommend(model, profile, top_k=5)

    assert spec.task_type == "tabular"
    assert spec.kernel_size is None
    assert 0.0 <= predicted_acc <= 1.0
    assert len(ranked) == 5
    # ranked must be sorted by predicted score, descending
    preds = [p for _, p in ranked]
    assert preds == sorted(preds, reverse=True)
