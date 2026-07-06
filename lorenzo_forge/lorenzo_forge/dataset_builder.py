"""Builds the meta-model's training corpus: for many random data profiles,
run an empirical architecture search and record (profile, winning spec) pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lorenzo_forge.search import search_best_architecture
from lorenzo_forge.synthetic import generate_dataset, sample_random_profile


def build_training_corpus(
    num_profiles: int = 80,
    candidates_per_profile: int = 8,
    search_epochs: int = 4,
    seed: int = 0,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for i in range(num_profiles):
        profile = sample_random_profile(rng)
        data = generate_dataset(profile, rng)
        best_spec, best_score, _ = search_best_architecture(
            profile, data, rng, num_candidates=candidates_per_profile, epochs=search_epochs
        )
        records.append(
            {
                "profile": profile.to_dict(),
                "best_spec": best_spec.to_raw_dict(),
                "score": best_score,
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
