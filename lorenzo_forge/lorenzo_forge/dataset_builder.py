"""Builds the meta-model's training corpus: for many random data profiles,
run an empirical architecture search and record every (candidate, score) trial
plus the tie-broken winner. The scorer meta-model learns from all trials."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lorenzo_forge.real_image import sample_real_image_profile
from lorenzo_forge.search import search_best_architecture
from lorenzo_forge.synthetic import generate_dataset, sample_random_profile, sample_tabular_profile


def build_training_corpus(
    num_profiles: int = 80,
    candidates_per_profile: int = 8,
    search_epochs: int = 4,
    seed: int = 0,
    out_path: str | Path | None = None,
    verbose: bool = True,
    real_images: bool = True,
    image_search_epochs: int | None = None,
) -> list[dict]:
    """Half tabular (synthetic) / half image profiles. With real_images=True,
    image profiles are drawn from real datasets (MNIST / Fashion-MNIST) where
    architecture choice actually matters; image candidates train for
    image_search_epochs (default: 2x search_epochs) since real images need more
    steps for a good arch to separate from a weak one."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    img_epochs = image_search_epochs if image_search_epochs is not None else search_epochs * 2

    for i in range(num_profiles):
        make_image = bool(rng.integers(0, 2))
        if make_image and real_images:
            profile, data = sample_real_image_profile(rng)
            epochs = img_epochs
        elif make_image:
            profile = sample_random_profile(rng)
            while profile.task_type != "image":
                profile = sample_random_profile(rng)
            data = generate_dataset(profile, rng)
            epochs = img_epochs
        else:
            profile = sample_tabular_profile(rng)
            data = generate_dataset(profile, rng)
            epochs = search_epochs

        best_spec, best_score, trials = search_best_architecture(
            profile, data, rng, num_candidates=candidates_per_profile, epochs=epochs
        )
        records.append(
            {
                "profile": profile.to_dict(),
                "best_spec": best_spec.to_raw_dict(),
                "score": best_score,
                "trials": [{"spec": s.to_raw_dict(), "score": sc} for s, sc in trials],
            }
        )
        if verbose:
            print(f"[{i + 1}/{num_profiles}] {profile.task_type} profile -> score={best_score:.3f} :: {best_spec.describe()}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def load_training_corpus(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
