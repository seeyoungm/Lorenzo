"""Random search over the architecture search space for one data profile.
The winner becomes the training label for the meta-model."""

from __future__ import annotations

import numpy as np

from lorenzo_forge.candidate_trainer import evaluate_spec
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec, sample_random_spec


def search_best_architecture(
    profile: DataProfile,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
    num_candidates: int = 8,
    epochs: int = 4,
) -> tuple[ArchitectureSpec, float, list[tuple[ArchitectureSpec, float]]]:
    trials: list[tuple[ArchitectureSpec, float]] = []
    best_spec: ArchitectureSpec | None = None
    best_score = -1.0

    seen: set[ArchitectureSpec] = set()
    attempts = 0
    while len(trials) < num_candidates and attempts < num_candidates * 3:
        attempts += 1
        spec = sample_random_spec(profile.task_type, rng)
        if spec in seen:
            continue
        seen.add(spec)
        score = evaluate_spec(spec, profile, data, epochs=epochs)
        trials.append((spec, score))
        if score > best_score:
            best_score, best_spec = score, spec

    assert best_spec is not None
    return best_spec, best_score, trials
