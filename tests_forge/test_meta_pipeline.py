from lorenzo_forge.dataset_builder import build_training_corpus
from lorenzo_forge.meta_model import build_meta_model, recommend, train_meta_model
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec, HEADS


def test_build_meta_model_output_heads():
    model = build_meta_model(input_dim=7)
    assert set(model.output_names) == set(HEADS.keys())


def test_corpus_and_meta_model_end_to_end():
    records = build_training_corpus(num_profiles=3, candidates_per_profile=2, search_epochs=1, seed=1, verbose=False)
    assert len(records) == 3
    for r in records:
        ArchitectureSpec.from_raw_dict(r["best_spec"])  # must not raise

    model = train_meta_model(records, epochs=2, verbose=0)
    profile = DataProfile("tabular", (10,), 3, 500, 1.0, 0.2)
    spec, confidences = recommend(model, profile)

    assert spec.task_type == "tabular"
    assert spec.kernel_size is None
    assert set(confidences.keys()) == set(HEADS.keys())
    assert all(0.0 <= v <= 1.0 for v in confidences.values())
