from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(slots=True)
class MemoryConfig:
    path: Path = Path("sample_data/memory_store.jsonl")
    top_k: int = 5
    retrieval_preselect_multiplier: int = 3
    min_similarity_floor: float = 0.10
    importance_weight: float = 0.10
    recency_weight: float = 0.15
    similarity_weight: float = 0.75
    lexical_fallback_weight: float = 0.05
    recency_half_life_hours: float = 72.0
    min_importance_to_store: float = 6.5
    dedup_similarity_threshold: float = 0.97
    semantic_merge_similarity_threshold: float = 0.60
    merge_confidence_threshold: float = 0.68


@dataclass(slots=True)
class AppConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    language_backend: str = "rule_based"

    @classmethod
    def from_toml(cls, path: str | Path) -> "AppConfig":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

        memory_raw = raw.get("memory", {})
        memory = MemoryConfig(
            path=Path(memory_raw.get("path", "sample_data/memory_store.jsonl")),
            top_k=int(memory_raw.get("top_k", 5)),
            retrieval_preselect_multiplier=int(memory_raw.get("retrieval_preselect_multiplier", 3)),
            min_similarity_floor=float(memory_raw.get("min_similarity_floor", 0.10)),
            importance_weight=float(memory_raw.get("importance_weight", 0.10)),
            recency_weight=float(memory_raw.get("recency_weight", 0.15)),
            similarity_weight=float(memory_raw.get("similarity_weight", 0.75)),
            lexical_fallback_weight=float(memory_raw.get("lexical_fallback_weight", 0.05)),
            recency_half_life_hours=float(memory_raw.get("recency_half_life_hours", 72.0)),
            min_importance_to_store=float(memory_raw.get("min_importance_to_store", 6.5)),
            dedup_similarity_threshold=float(memory_raw.get("dedup_similarity_threshold", 0.97)),
            semantic_merge_similarity_threshold=float(
                memory_raw.get("semantic_merge_similarity_threshold", 0.60)
            ),
            merge_confidence_threshold=float(memory_raw.get("merge_confidence_threshold", 0.68)),
        )

        language_backend = raw.get("language", {}).get("backend", "rule_based")
        return cls(memory=memory, language_backend=language_backend)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    return AppConfig.from_toml(path)
