from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import math
import re

from lorenzo.memory.embedding import MultilingualEmbeddingEncoder
from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


class MemoryRetriever:
    def __init__(
        self,
        importance_weight: float = 0.10,
        recency_weight: float = 0.15,
        similarity_weight: float = 0.75,
        lexical_fallback_weight: float = 0.05,
        recency_half_life_hours: float = 72.0,
        retrieval_preselect_multiplier: int = 3,
        min_similarity_floor: float = 0.10,
        type_alignment_weight: float = 0.10,
        access_frequency_weight: float = 0.05,
        conflict_penalty_weight: float = 0.12,
        stale_conflict_hard_penalty: float = 0.55,
        diversity_penalty_weight: float = 0.10,
        diversity_similarity_threshold: float = 0.86,
    ) -> None:
        self.similarity_weight = max(0.0, similarity_weight)
        self.recency_weight = max(0.0, recency_weight)
        self.importance_weight = max(0.0, importance_weight)
        self.lexical_fallback_weight = max(0.0, lexical_fallback_weight)
        self.type_alignment_weight = max(0.0, type_alignment_weight)
        self.access_frequency_weight = max(0.0, access_frequency_weight)
        self.conflict_penalty_weight = max(0.0, conflict_penalty_weight)
        self.stale_conflict_hard_penalty = max(0.0, stale_conflict_hard_penalty)
        self.diversity_penalty_weight = max(0.0, diversity_penalty_weight)
        self.diversity_similarity_threshold = max(0.0, min(1.0, diversity_similarity_threshold))

        total = (
            self.similarity_weight
            + self.recency_weight
            + self.importance_weight
            + self.lexical_fallback_weight
            + self.type_alignment_weight
            + self.access_frequency_weight
        )
        if total <= 0:
            raise ValueError("At least one retrieval weight must be positive")

        self.similarity_weight /= total
        self.recency_weight /= total
        self.importance_weight /= total
        self.lexical_fallback_weight /= total
        self.type_alignment_weight /= total
        self.access_frequency_weight /= total
        self._enforce_semantic_dominance(min_semantic_weight=0.60)

        self.recency_half_life_hours = recency_half_life_hours
        self.retrieval_preselect_multiplier = max(1, retrieval_preselect_multiplier)
        self.min_similarity_floor = max(0.0, min(1.0, min_similarity_floor))

        self.encoder = MultilingualEmbeddingEncoder()
        self._memory_embedding_cache: dict[str, tuple[str, list[float]]] = {}
        self._query_embedding_cache: dict[str, list[float]] = {}

    def retrieve(
        self,
        query: str,
        memories: list[MemoryItem],
        top_k: int = 5,
        now: datetime | None = None,
    ) -> list[RetrievedMemory]:
        now = now or datetime.now(timezone.utc)
        if not memories or top_k <= 0:
            return []

        intent = self._infer_intent(query)
        query_embedding = self._get_query_embedding(query)
        stale_conflict_flags = self._stale_conflict_flags(memories)

        semantic_raw: list[float] = []
        lexical_raw: list[float] = []
        recency_raw: list[float] = []
        importance_raw: list[float] = []
        type_alignment_raw: list[float] = []
        access_raw: list[float] = []
        conflict_penalty_raw: list[float] = []

        for idx, memory in enumerate(memories):
            memory_embedding = self._get_memory_embedding(memory)
            semantic_score = self._semantic_similarity(
                query,
                memory.content,
                query_embedding=query_embedding,
                memory_embedding=memory_embedding,
            )
            lexical_score = self._lexical_overlap(query, memory.content)
            recency_score = self._recency(memory.created_at, now)
            importance_score = max(0.0, min(1.0, memory.importance / 10.0))
            type_alignment = self._type_alignment_score(intent, memory)
            access_score = self._access_frequency_score(memory)
            conflict_penalty = stale_conflict_flags[idx]

            semantic_raw.append(semantic_score)
            lexical_raw.append(lexical_score)
            recency_raw.append(recency_score)
            importance_raw.append(importance_score)
            type_alignment_raw.append(type_alignment)
            access_raw.append(access_score)
            conflict_penalty_raw.append(conflict_penalty)

        semantic_norm = self._normalize(semantic_raw)
        lexical_norm = self._normalize(lexical_raw)
        recency_norm = self._normalize(recency_raw)
        importance_norm = self._normalize(importance_raw)
        type_alignment_norm = self._normalize(type_alignment_raw)
        access_norm = self._normalize(access_raw)
        conflict_norm = self._normalize(conflict_penalty_raw)

        scored: list[RetrievedMemory] = []
        for idx, memory in enumerate(memories):
            fallback_lexical = lexical_norm[idx] if semantic_raw[idx] < self.min_similarity_floor else 0.0
            relevance_penalty = -0.12 if semantic_raw[idx] < self.min_similarity_floor else 0.0

            total_score = (
                semantic_norm[idx] * self.similarity_weight
                + type_alignment_norm[idx] * self.type_alignment_weight
                + recency_norm[idx] * self.recency_weight
                + importance_norm[idx] * self.importance_weight
                + access_norm[idx] * self.access_frequency_weight
                + fallback_lexical * self.lexical_fallback_weight
                - conflict_norm[idx] * self.conflict_penalty_weight
                - conflict_penalty_raw[idx] * self.stale_conflict_hard_penalty
                - self._working_context_penalty(intent, memory)
                + relevance_penalty
            )
            reason = (
                f"semantic(raw={semantic_raw[idx]:.3f}, norm={semantic_norm[idx]:.3f}); "
                f"type_alignment(norm={type_alignment_norm[idx]:.3f}, intent={intent}); "
                f"recency(norm={recency_norm[idx]:.3f}); "
                f"importance(norm={importance_norm[idx]:.3f}); "
                f"access(norm={access_norm[idx]:.3f}); "
                f"lexical_fallback(norm={fallback_lexical:.3f}); "
                f"conflict_penalty(norm={conflict_norm[idx]:.3f}, hard={conflict_penalty_raw[idx]*self.stale_conflict_hard_penalty:.3f}); "
                f"working_context_penalty={self._working_context_penalty(intent, memory):.3f}; "
                f"penalty={relevance_penalty:.3f}"
            )
            scored.append(
                RetrievedMemory(
                    memory=memory,
                    total_score=total_score,
                    similarity_score=semantic_raw[idx],
                    recency_score=recency_raw[idx],
                    importance_score=importance_raw[idx],
                    retrieval_reason=reason,
                )
            )

        scored.sort(key=lambda item: item.total_score, reverse=True)
        preselect_size = max(top_k, top_k * self.retrieval_preselect_multiplier)
        preselected = scored[: max(preselect_size, 0)]
        reranked = self._apply_diversity_penalty(preselected)
        reranked.sort(key=lambda item: item.total_score, reverse=True)
        return reranked[: max(top_k, 0)]

    def text_similarity(self, left: str, right: str) -> float:
        left_embedding = self._get_query_embedding(left)
        right_embedding = self._get_query_embedding(right)
        return self._semantic_similarity(
            left,
            right,
            query_embedding=left_embedding,
            memory_embedding=right_embedding,
        )

    def _get_memory_embedding(self, memory: MemoryItem) -> list[float]:
        signature = sha1(memory.content.encode("utf-8")).hexdigest()
        cached = self._memory_embedding_cache.get(memory.memory_id)
        if cached and cached[0] == signature:
            return cached[1]

        embedding = self.encoder.encode(memory.content)
        self._memory_embedding_cache[memory.memory_id] = (signature, embedding)
        return embedding

    def _get_query_embedding(self, text: str) -> list[float]:
        key = self.encoder.cache_key(text)
        cached = self._query_embedding_cache.get(key)
        if cached is not None:
            return cached

        embedding = self.encoder.encode(text)
        self._query_embedding_cache[key] = embedding
        return embedding

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if math.isclose(lo, hi, abs_tol=1e-9):
            return [0.5 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]

    def _enforce_semantic_dominance(self, min_semantic_weight: float) -> None:
        if self.similarity_weight >= min_semantic_weight:
            return

        remainder = (
            self.recency_weight
            + self.importance_weight
            + self.lexical_fallback_weight
            + self.type_alignment_weight
            + self.access_frequency_weight
        )
        if remainder <= 0:
            self.similarity_weight = 1.0
            self.recency_weight = 0.0
            self.importance_weight = 0.0
            self.lexical_fallback_weight = 0.0
            self.type_alignment_weight = 0.0
            self.access_frequency_weight = 0.0
            return

        target_remainder = 1.0 - min_semantic_weight
        scale = target_remainder / remainder
        self.recency_weight *= scale
        self.importance_weight *= scale
        self.lexical_fallback_weight *= scale
        self.type_alignment_weight *= scale
        self.access_frequency_weight *= scale
        self.similarity_weight = min_semantic_weight

    def _semantic_similarity(
        self,
        left: str,
        right: str,
        query_embedding: list[float],
        memory_embedding: list[float],
    ) -> float:
        embed_cosine = self.encoder.cosine(query_embedding, memory_embedding)
        concept_overlap = self._concept_overlap(left, right)
        return (0.85 * embed_cosine) + (0.15 * concept_overlap)

    def _lexical_overlap(self, query: str, document: str) -> float:
        q_set = set(self._tokenize(query))
        d_set = set(self._tokenize(document))
        if not q_set or not d_set:
            return 0.0
        union = q_set | d_set
        if not union:
            return 0.0
        return len(q_set & d_set) / len(union)

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _concept_overlap(self, left: str, right: str) -> float:
        left_set = set(self.encoder.canonical_tokens(left))
        right_set = set(self.encoder.canonical_tokens(right))
        if not left_set or not right_set:
            return 0.0
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    def _recency(self, created_at: datetime, now: datetime) -> float:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (now - created_at).total_seconds())
        half_life_seconds = max(1.0, self.recency_half_life_hours * 3600.0)
        # Soft non-linear decay (slower early decay, steeper long-tail decay).
        return math.exp(-math.sqrt(age_seconds / half_life_seconds))

    def _infer_intent(self, query: str) -> str:
        q = query.lower().strip()
        is_question = q.endswith("?")
        if any(token in q for token in ["goal", "목표", "want", "만들고 싶", "하고 싶"]):
            return "goal"
        if any(token in q for token in ["preference", "prefer", "선호", "좋아", "싫어"]):
            return "preference"
        recall_hit = any(
            token in q
            for token in [
                "remember",
                "기억해",
                "기억하",
                "이전에",
                "before",
                "recall",
                "what did i",
            ]
        ) or (is_question and any(token in q for token in ["what is my", "what was my", "뭐였", "얼마", "몇 시", "최종", "latest", "current", "현재"]))
        if recall_hit:
            if any(token in q for token in ["meeting", "appointment", "회의", "일정", "time", "시간"]):
                return "memory_recall_event"
            if any(token in q for token in ["budget", "예산", "fact", "deadline", "마감", "마감일", "핵심 사실"]):
                return "memory_recall_fact"
            return "memory_recall"
        if any(token in q for token in ["fact", "사실", "is ", "입니다", "였다"]):
            return "fact"
        if q.endswith("?"):
            return "question"
        return "general"

    def _type_alignment_score(self, intent: str, memory: MemoryItem) -> float:
        tags = {tag.lower() for tag in memory.tags}
        memory_type = memory.memory_type

        if intent == "goal":
            if memory_type is MemoryType.SEMANTIC and "goal" in tags:
                return 1.0
            if memory_type is MemoryType.SEMANTIC:
                return 0.8
            return 0.2

        if intent == "preference":
            if memory_type is MemoryType.SEMANTIC and "preference" in tags:
                return 1.0
            if memory_type is MemoryType.SEMANTIC:
                return 0.25
            return 0.2

        if intent == "fact":
            if memory_type is MemoryType.SEMANTIC and "fact" in tags:
                return 1.0
            if memory_type is MemoryType.SEMANTIC:
                return 0.20
            return 0.2

        if intent == "memory_recall_event":
            if memory_type is MemoryType.EPISODIC and "event" in tags:
                return 1.0
            if memory_type is MemoryType.EPISODIC:
                return 0.25
            if memory_type is MemoryType.SEMANTIC:
                if "event" in tags:
                    return 0.65
                return 0.15
            return 0.05

        if intent == "memory_recall_fact":
            if memory_type is MemoryType.SEMANTIC:
                if "fact" in tags:
                    return 1.0
                return 0.20
            if memory_type is MemoryType.EPISODIC:
                return 0.10
            return 0.05

        if intent == "memory_recall":
            if memory_type is MemoryType.SEMANTIC:
                if "fact" in tags or "preference" in tags or "goal" in tags or "commitment" in tags:
                    return 1.0
                return 0.50
            if memory_type is MemoryType.EPISODIC:
                if "event" in tags:
                    return 0.75
                return 0.35
            return 0.05

        if intent == "question":
            if memory_type is MemoryType.SEMANTIC:
                return 0.60
            if memory_type is MemoryType.WORKING:
                return 0.25
            return 0.35

        if memory_type is MemoryType.SEMANTIC:
            return 0.55
        if memory_type is MemoryType.EPISODIC:
            return 0.45
        return 0.35

    def _access_frequency_score(self, memory: MemoryItem) -> float:
        access_count = int(memory.metadata.get("access_count", 0))
        return math.log1p(max(0, access_count))

    def _stale_conflict_flags(self, memories: list[MemoryItem]) -> list[float]:
        flags = [0.0 for _ in memories]
        groups: dict[str, list[tuple[int, str, datetime]]] = {}

        for idx, memory in enumerate(memories):
            key, value = "", ""
            if memory.memory_type is MemoryType.SEMANTIC:
                key, value = self._extract_fact_kv(memory.content)
            elif memory.memory_type is MemoryType.EPISODIC:
                key, value = self._extract_event_kv(memory.content)
            if not key or not value:
                continue
            created = memory.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            groups.setdefault(key, []).append((idx, value, created))

        for entries in groups.values():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda item: item[2])
            latest_index, latest_value, _ = entries[-1]
            for idx, value, _ in entries[:-1]:
                if value != latest_value:
                    flags[idx] = 1.0
            flags[latest_index] = 0.0

        return flags

    def _extract_event_kv(self, content: str) -> tuple[str, str]:
        normalized = content.replace("User event:", "").strip().lower()
        kr = re.search(r"(회의|미팅).{0,24}?(\d{1,2}\s*시)", normalized)
        if kr:
            return "meeting_time", kr.group(2).replace(" ", "")
        en = re.search(r"(meeting|appointment).{0,32}?(\d{1,2}:\d{2}\s*(am|pm)?)", normalized)
        if en:
            return "meeting_time", en.group(2).replace(" ", "")
        return "", ""

    def _working_context_penalty(self, intent: str, memory: MemoryItem) -> float:
        if memory.memory_type is not MemoryType.WORKING:
            return 0.0
        if memory.metadata.get("policy") != "question_working_only":
            return 0.0
        if intent.startswith("memory_recall"):
            return 0.85
        if intent in {"goal", "fact", "preference"}:
            return 0.35
        return 0.0

    def _extract_fact_kv(self, content: str) -> tuple[str, str]:
        normalized = content.replace("User fact:", "").strip()
        kr = re.search(r"(.+?)(은|는|이|가)\s*(.+?)(입니다|이다|였어|였지|$)", normalized)
        if kr:
            key = kr.group(1).strip().lower()
            value = kr.group(3).strip().lower()
            return key, value

        en = re.search(r"(.+?)\s+is\s+(.+)", normalized.lower())
        if en:
            key = en.group(1).strip()
            value = en.group(2).strip()
            return key, value
        return "", ""

    def _apply_diversity_penalty(self, items: list[RetrievedMemory]) -> list[RetrievedMemory]:
        if len(items) <= 1 or self.diversity_penalty_weight <= 0:
            return items

        remaining = list(items)
        selected: list[RetrievedMemory] = []
        scale = max(1e-9, 1.0 - self.diversity_similarity_threshold)

        while remaining:
            best_index = 0
            best_adjusted_score = -10**9
            best_penalty = 0.0
            best_similarity = 0.0

            for idx, item in enumerate(remaining):
                max_similarity = 0.0
                if selected:
                    max_similarity = max(
                        self.text_similarity(item.memory.content, chosen.memory.content) for chosen in selected
                    )
                over = max(0.0, max_similarity - self.diversity_similarity_threshold)
                diversity_penalty = self.diversity_penalty_weight * (over / scale)
                adjusted_score = item.total_score - diversity_penalty
                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_index = idx
                    best_penalty = diversity_penalty
                    best_similarity = max_similarity

            chosen = remaining.pop(best_index)
            chosen.total_score = best_adjusted_score
            chosen.retrieval_reason = (
                f"{chosen.retrieval_reason}; diversity(max_sim={best_similarity:.3f}, penalty={best_penalty:.3f})"
            )
            selected.append(chosen)

        return selected
