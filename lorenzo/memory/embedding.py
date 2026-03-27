from __future__ import annotations

from hashlib import sha1
import math
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


class MultilingualEmbeddingEncoder:
    """Lightweight multilingual embedding encoder without external dependencies.

    It normalizes multilingual domain terms into canonical concepts, then
    builds a hashed vector using token and char-ngram features.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._concept_map = self._build_concept_map()

    def encode(self, text: str) -> list[float]:
        canonical_tokens = self.canonical_tokens(text)

        vec = [0.0] * self.dim
        for token in canonical_tokens:
            self._accumulate_feature(vec, f"tok:{token}", 1.0)
            for ngram in self._char_ngrams(token):
                self._accumulate_feature(vec, f"ng:{ngram}", 0.35)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))

    def cache_key(self, text: str) -> str:
        return sha1(text.encode("utf-8")).hexdigest()

    def canonical_tokens(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [self._concept_map.get(token, token) for token in tokens]

    def _accumulate_feature(self, vec: list[float], feature: str, weight: float) -> None:
        idx = int(sha1(feature.encode("utf-8")).hexdigest(), 16) % self.dim
        vec[idx] += weight

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _char_ngrams(self, token: str) -> list[str]:
        padded = f"_{token}_"
        ngrams: list[str] = []
        for n in (2, 3):
            if len(padded) < n:
                continue
            ngrams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return ngrams

    def _build_concept_map(self) -> dict[str, str]:
        synonyms = {
            "goal": ["goal", "목표", "goals"],
            "memory": ["memory", "기억", "remember", "저장", "save"],
            "long_term": ["장기", "long", "longterm", "long-term"],
            "module": ["module", "modular", "모듈"],
            "reasoning": ["reasoning", "추론"],
            "language": ["language", "언어"],
            "preference": ["preference", "prefer", "선호"],
            "commitment": ["commitment", "commit", "약속", "할게", "하겠다"],
            "event": ["event", "meeting", "appointment", "일정", "회의", "약속", "마감"],
            "fact": ["fact", "사실"],
            "deadline": ["deadline", "마감"],
            "architecture": ["architecture", "구조", "아키텍처"],
            "persistent": ["persistent", "영속", "지속"],
            "working": ["working", "워킹"],
            "episodic": ["episodic", "에피소드"],
            "semantic": ["semantic", "시맨틱"],
        }

        concept_map: dict[str, str] = {}
        for concept, words in synonyms.items():
            for word in words:
                concept_map[word.lower()] = concept
        return concept_map
