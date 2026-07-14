import numpy as np

from lorenzo_forge.dataset_builder import build_training_corpus, load_training_corpus
from lorenzo_forge.meta_model import SCORER_INPUT_DIM, build_scorer_model, recommend, train_scorer_model
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec, enumerate_specs


def test_scorer_model_output_is_scalar():
    model = build_scorer_model()
    assert model.output_shape == (None, 1)
    assert model.input_shape == (None, SCORER_INPUT_DIM)


def test_enumerate_specs_sizes():
    # x2 x2 at the end = BLOCK_STYLE_CHOICES x POOL_STYLE_CHOICES (Track 2 modernization axes).
    # tabular has no constraint applied (deep-recurrent lr cap & image-residual
    # unit cap don't touch it), so it stays the full product.
    assert len(list(enumerate_specs("tabular"))) == 4 * 4 * 1 * 2 * 3 * 2 * 3 * 2 * 2
    # image: full 4608 minus 576 residual+units=256 combos excluded by is_valid_spec.
    assert len(list(enumerate_specs("image"))) == 4 * 4 * 2 * 2 * 3 * 2 * 3 * 2 * 2 - 576


def test_corpus_records_include_all_trials():
    records = build_training_corpus(
        num_profiles=2, candidates_per_profile=3, search_epochs=1, seed=2, verbose=False, real_images=False, domains=("tabular",)
    )
    for r in records:
        assert len(r["trials"]) >= 1
        for t in r["trials"]:
            ArchitectureSpec.from_raw_dict(t["spec"])
            assert 0.0 <= t["score"] <= 1.0


def test_incremental_corpus_rebuild_only_touches_target_domain(tmp_path):
    base_path = tmp_path / "base.jsonl"
    build_training_corpus(
        num_profiles=6, candidates_per_profile=3, search_epochs=1, seed=3, verbose=False,
        real_images=False, domains=("tabular", "timeseries"), out_path=base_path,
    )
    base_records = load_training_corpus(base_path)
    base_tabular = [r for r in base_records if r["profile"]["task_type"] == "tabular"]
    assert base_tabular  # sanity: the domain we're about to leave untouched exists

    rebuilt = build_training_corpus(
        num_profiles=4, candidates_per_profile=3, search_epochs=1, seed=4, verbose=False,
        real_images=False, domains=("timeseries",), base_corpus_path=base_path,
    )

    rebuilt_tabular = [r for r in rebuilt if r["profile"]["task_type"] == "tabular"]
    rebuilt_timeseries = [r for r in rebuilt if r["profile"]["task_type"] == "timeseries"]
    # untouched domain carried over byte-for-byte from the base corpus
    assert rebuilt_tabular == base_tabular
    # touched domain was freshly (re)searched at the new profile count
    assert len(rebuilt_timeseries) == 4
    assert len(rebuilt) == len(base_tabular) + 4


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
    # ranked is ordered by score bucket (descending) then complexity, so it is
    # descending up to within-bucket tie-break: no pick may sit more than one
    # tie_tolerance below an earlier one.
    tie_tolerance = 0.02
    preds = [p for _, p in ranked]
    for earlier, later in zip(preds, preds[1:]):
        assert later <= earlier + tie_tolerance


class _ConstantScorer:
    """Stand-in scorer that predicts the same accuracy for every architecture,
    forcing the entire search space into a single tie bucket."""

    def predict(self, feats, verbose=0):
        return np.full((len(feats), 1), 0.5, dtype="float32")


def test_recommend_tie_break_prefers_cheapest_when_scores_tie():
    profile = DataProfile("tabular", (10,), 3, 500, 1.0, 0.2)
    specs = list(enumerate_specs(profile.task_type))
    cheapest = min(specs, key=lambda s: (s.complexity_proxy(), s.describe()))

    spec, predicted_acc, ranked = recommend(_ConstantScorer(), profile, top_k=5)

    # Every architecture scores identically, so the complexity tie-break decides:
    # the top pick must be the single cheapest spec in the space.
    assert spec == cheapest
    assert predicted_acc == 0.5
    proxies = [s.complexity_proxy() for s, _ in ranked]
    assert proxies == sorted(proxies)  # ranked ascending in cost within the tie
