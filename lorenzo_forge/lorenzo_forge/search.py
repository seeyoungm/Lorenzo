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
    tie_tolerance: float = 0.02,
) -> tuple[ArchitectureSpec, float, list[tuple[ArchitectureSpec, float]]]:
    trials: list[tuple[ArchitectureSpec, float]] = []

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

    assert trials
    top_score = max(score for _, score in trials)
    # Deterministic tie-break: among candidates within tie_tolerance of the top
    # score, prefer the simplest one. Without this, when many architectures tie
    # (common for easy tasks) the recorded winner is arbitrary -> label noise.
    contenders = [(s, sc) for s, sc in trials if top_score - sc <= tie_tolerance]
    best_spec, _ = min(contenders, key=lambda pair: (pair[0].complexity_proxy(), pair[0].describe()))
    best_score = dict((s, sc) for s, sc in trials)[best_spec]

    return best_spec, best_score, trials
