from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

from lorenzo.config import AppConfig
from lorenzo.input_processor import InputProcessor
from lorenzo.interfaces import (
    InputProcessorPort,
    LanguageAdapterPort,
    MemoryRetrieverPort,
    MemoryStorePort,
    ReasoningPlannerPort,
)
from lorenzo.language import LanguageAdapter
from lorenzo.memory import JsonlMemoryStore, MemoryRetriever
from lorenzo.models import InputType, MemoryItem, MemoryType, ProcessedInput, TurnResult
from lorenzo.reasoning import ReasoningPlanner


@dataclass(slots=True)
class PipelineModules:
    input_processor: InputProcessorPort
    memory_store: MemoryStorePort
    memory_retriever: MemoryRetrieverPort
    reasoning_planner: ReasoningPlannerPort
    language_adapter: LanguageAdapterPort


@dataclass(slots=True)
class UpdateTelemetry:
    turns: int = 0
    candidates_seen: int = 0
    merge_attempts: int = 0
    merge_success_count: int = 0
    merge_rejected_count: int = 0
    merge_applied: int = 0
    false_merge_attempts: int = 0
    conflict_resolved: int = 0


class LorenzoOrchestrator:
    def __init__(
        self,
        modules: PipelineModules,
        top_k: int = 5,
        min_importance_to_store: float = 6.0,
        dedup_similarity_threshold: float = 0.92,
        semantic_merge_similarity_threshold: float = 0.78,
        merge_confidence_threshold: float = 0.82,
    ) -> None:
        self.modules = modules
        self.top_k = top_k
        self.min_importance_to_store = min_importance_to_store
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.semantic_merge_similarity_threshold = semantic_merge_similarity_threshold
        self.merge_confidence_threshold = merge_confidence_threshold
        self._telemetry = UpdateTelemetry()

    @classmethod
    def from_config(cls, config: AppConfig) -> "LorenzoOrchestrator":
        modules = PipelineModules(
            input_processor=InputProcessor(),
            memory_store=JsonlMemoryStore(config.memory.path),
            memory_retriever=MemoryRetriever(
                importance_weight=config.memory.importance_weight,
                recency_weight=config.memory.recency_weight,
                similarity_weight=config.memory.similarity_weight,
                lexical_fallback_weight=config.memory.lexical_fallback_weight,
                recency_half_life_hours=config.memory.recency_half_life_hours,
                retrieval_preselect_multiplier=config.memory.retrieval_preselect_multiplier,
                min_similarity_floor=config.memory.min_similarity_floor,
            ),
            reasoning_planner=ReasoningPlanner(),
            language_adapter=LanguageAdapter.from_name(config.language_backend),
        )
        return cls(
            modules=modules,
            top_k=config.memory.top_k,
            min_importance_to_store=config.memory.min_importance_to_store,
            dedup_similarity_threshold=config.memory.dedup_similarity_threshold,
            semantic_merge_similarity_threshold=config.memory.semantic_merge_similarity_threshold,
            merge_confidence_threshold=config.memory.merge_confidence_threshold,
        )

    def run_turn(self, user_input: str) -> TurnResult:
        processed = self.modules.input_processor.process(user_input)

        existing_memories = self.modules.memory_store.list_all()
        retrieved = self.modules.memory_retriever.retrieve(
            query=processed.raw_text,
            memories=existing_memories,
            top_k=self.top_k,
        )
        if self._apply_retrieval_feedback(existing_memories, retrieved):
            self.modules.memory_store.replace_all(existing_memories)

        plan = self.modules.reasoning_planner.plan(processed, retrieved)
        response = self.modules.language_adapter.generate(
            user_input=processed.raw_text,
            processed=processed,
            retrieved=retrieved,
            plan=plan,
        )

        self._update_memories(processed)
        return TurnResult(
            user_input=processed.raw_text,
            response=response,
            retrieved_memories=retrieved,
            plan=plan,
        )

    def snapshot_telemetry(self) -> UpdateTelemetry:
        return UpdateTelemetry(
            turns=self._telemetry.turns,
            candidates_seen=self._telemetry.candidates_seen,
            merge_attempts=self._telemetry.merge_attempts,
            merge_success_count=self._telemetry.merge_success_count,
            merge_rejected_count=self._telemetry.merge_rejected_count,
            merge_applied=self._telemetry.merge_applied,
            false_merge_attempts=self._telemetry.false_merge_attempts,
            conflict_resolved=self._telemetry.conflict_resolved,
        )

    def _update_memories(self, processed: ProcessedInput) -> None:
        now = datetime.now(timezone.utc)
        existing = self.modules.memory_store.list_all()
        changed = False
        self._telemetry.turns += 1
        candidates = self._build_memory_candidates(processed, now)
        self._telemetry.candidates_seen += len(candidates)

        for candidate in candidates:
            is_working_question = candidate.metadata.get("policy") == "question_working_only"
            if not is_working_question and candidate.importance < self.min_importance_to_store:
                continue

            match_index, match_similarity = self._find_best_match(existing, candidate)
            if match_index >= 0 and match_similarity >= self.dedup_similarity_threshold:
                self._merge_duplicate(existing[match_index], candidate, match_similarity, now)
                changed = True
                continue

            if (
                candidate.memory_type is MemoryType.SEMANTIC
                and match_index >= 0
                and self._should_resolve_conflict(existing[match_index], candidate)
            ):
                self._resolve_conflict_latest_wins(existing[match_index], candidate, now)
                self._telemetry.conflict_resolved += 1
                changed = True
                continue

            if (
                candidate.memory_type is MemoryType.EPISODIC
                and match_index >= 0
                and self._should_resolve_event_conflict(existing[match_index], candidate)
            ):
                self._resolve_conflict_latest_wins(existing[match_index], candidate, now)
                self._telemetry.conflict_resolved += 1
                changed = True
                continue

            if (
                candidate.memory_type is MemoryType.SEMANTIC
                and match_index >= 0
                and match_similarity >= self.semantic_merge_similarity_threshold
            ):
                self._telemetry.merge_attempts += 1
                confidence = self._merge_confidence(existing[match_index], candidate, match_similarity)
                if confidence < self.merge_confidence_threshold:
                    self._telemetry.merge_rejected_count += 1
                    existing.append(candidate)
                    changed = True
                    continue

                if self._is_false_merge_candidate(existing[match_index], candidate):
                    self._telemetry.merge_rejected_count += 1
                    existing.append(candidate)
                    changed = True
                    continue
                merged = self._synthesize_semantic_memory(
                    existing=existing[match_index],
                    candidate=candidate,
                    similarity=match_similarity,
                    confidence=confidence,
                    now=now,
                )
                existing[match_index] = merged
                self._telemetry.merge_success_count += 1
                self._telemetry.merge_applied += 1
                changed = True
                continue

            existing.append(candidate)
            changed = True

        if changed:
            self.modules.memory_store.replace_all(existing)

    def _apply_retrieval_feedback(
        self, existing_memories: list[MemoryItem], retrieved: list
    ) -> bool:
        if not existing_memories or not retrieved:
            return False

        memory_by_id = {item.memory_id: item for item in existing_memories}
        now = datetime.now(timezone.utc).isoformat()
        changed = False
        for ranked, item in enumerate(retrieved, start=1):
            memory = memory_by_id.get(item.memory.memory_id)
            if memory is None:
                continue
            access_count = int(memory.metadata.get("access_count", 0)) + 1
            memory.metadata["access_count"] = access_count
            memory.metadata["last_accessed_at"] = now
            memory.metadata["last_rank"] = ranked
            changed = True
        return changed

    def _build_memory_candidates(self, processed: ProcessedInput, now: datetime) -> list[MemoryItem]:
        text = processed.raw_text
        labels = set(processed.input_types)
        base_importance = self._estimate_importance(text, labels)

        # Policy:
        # QUESTION -> working only
        # GOAL/FACT/PREFERENCE/COMMITMENT -> semantic
        # EVENT -> episodic
        # Store only meaningful categories.
        candidates: list[MemoryItem] = []

        question_only = (
            InputType.QUESTION in labels
            and InputType.MEMORY_CANDIDATE not in labels
            and InputType.MEMORY_INSTRUCTION not in labels
            and InputType.GOAL_STATEMENT not in labels
            and InputType.PREFERENCE not in labels
            and InputType.COMMITMENT not in labels
            and InputType.FACT not in labels
        )
        if question_only:
            candidates.append(
                MemoryItem(
                    content=f"Working question context: {text}",
                    memory_type=MemoryType.WORKING,
                    importance=max(6.2, base_importance),
                    metadata={"source": "user", "policy": "question_working_only"},
                    tags=["working", "question"],
                    created_at=now,
                )
            )
            return candidates

        if InputType.GOAL_STATEMENT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User goal: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(8.0, base_importance),
                    metadata={"rule": "goal_to_semantic"},
                    tags=["goal", "semantic"],
                    created_at=now,
                )
            )

        if InputType.PREFERENCE in labels:
            candidates.append(
                MemoryItem(
                    content=f"User preference: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.0, base_importance),
                    metadata={"rule": "preference_to_semantic"},
                    tags=["preference", "semantic"],
                    created_at=now,
                )
            )

        if InputType.COMMITMENT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User commitment: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.5, base_importance),
                    metadata={"rule": "commitment_to_semantic"},
                    tags=["commitment", "semantic"],
                    created_at=now,
                )
            )

        if InputType.FACT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User fact: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.2, base_importance),
                    metadata={"rule": "fact_to_semantic"},
                    tags=["fact", "semantic"],
                    created_at=now,
                )
            )

        if InputType.EVENT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User event: {text}",
                    memory_type=MemoryType.EPISODIC,
                    importance=max(6.5, base_importance),
                    metadata={"rule": "event_to_episodic"},
                    tags=["event", "episodic"],
                    created_at=now,
                )
            )

        # If no allowed persistent type matched, skip storage.
        return candidates

    def _estimate_importance(self, text: str, labels: set[InputType]) -> float:
        score = 5.0
        if InputType.GOAL_STATEMENT in labels:
            score = max(score, 8.5)
        if InputType.COMMITMENT in labels:
            score = max(score, 8.0)
        if InputType.PREFERENCE in labels:
            score = max(score, 7.4)
        if InputType.FACT in labels:
            score = max(score, 7.2)
        if InputType.EVENT in labels:
            score = max(score, 6.8)
        question_only = (
            InputType.QUESTION in labels
            and InputType.MEMORY_CANDIDATE not in labels
            and InputType.MEMORY_INSTRUCTION not in labels
            and InputType.GOAL_STATEMENT not in labels
            and InputType.PREFERENCE not in labels
            and InputType.COMMITMENT not in labels
            and InputType.FACT not in labels
        )
        if question_only:
            score = 5.8

        lowered = text.lower()
        if any(token in lowered for token in ["important", "중요", "반드시", "must", "deadline", "마감"]):
            score += 0.6
        if len(text.strip()) <= 8:
            score -= 0.8
        return max(0.0, min(10.0, score))

    def _should_resolve_conflict(self, existing: MemoryItem, candidate: MemoryItem) -> bool:
        existing_tags = {tag.lower() for tag in existing.tags}
        candidate_tags = {tag.lower() for tag in candidate.tags}
        if "fact" not in existing_tags or "fact" not in candidate_tags:
            return False

        existing_key, existing_value = self._extract_fact_kv(existing.content)
        candidate_key, candidate_value = self._extract_fact_kv(candidate.content)
        if not existing_key or not candidate_key:
            return False

        key_similarity = self.modules.memory_retriever.text_similarity(existing_key, candidate_key)
        if key_similarity < 0.70:
            return False
        if existing_value == candidate_value:
            return False
        return True

    def _resolve_conflict_latest_wins(self, existing: MemoryItem, candidate: MemoryItem, now: datetime) -> None:
        history = list(existing.metadata.get("conflict_history", []))
        history.append(
            {
                "at": now.isoformat(),
                "previous": existing.content,
                "incoming": candidate.content,
            }
        )
        existing.content = candidate.content
        existing.importance = min(10.0, max(existing.importance, candidate.importance))
        existing.metadata["conflict_policy"] = "latest_wins"
        existing.metadata["conflict_history"] = history[-20:]
        existing.metadata["last_conflict_at"] = now.isoformat()
        existing.tags = sorted(set(existing.tags) | set(candidate.tags) | {"conflict_resolved"})

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

    def _should_resolve_event_conflict(self, existing: MemoryItem, candidate: MemoryItem) -> bool:
        existing_tags = {tag.lower() for tag in existing.tags}
        candidate_tags = {tag.lower() for tag in candidate.tags}
        if "event" not in existing_tags or "event" not in candidate_tags:
            return False

        existing_key, existing_value = self._extract_event_kv(existing.content)
        candidate_key, candidate_value = self._extract_event_kv(candidate.content)
        if not existing_key or not candidate_key:
            return False
        if existing_key != candidate_key:
            return False
        if existing_value == candidate_value:
            return False
        return True

    def _extract_event_kv(self, content: str) -> tuple[str, str]:
        normalized = content.replace("User event:", "").strip().lower()
        kr = re.search(r"(회의|미팅).{0,24}?(\d{1,2}\s*시)", normalized)
        if kr:
            return "meeting_time", kr.group(2).replace(" ", "")
        en = re.search(r"(meeting|appointment).{0,32}?(\d{1,2}:\d{2}\s*(am|pm)?)", normalized)
        if en:
            return "meeting_time", en.group(2).replace(" ", "")
        return "", ""

    def _is_false_merge_candidate(self, existing: MemoryItem, candidate: MemoryItem) -> bool:
        if self._should_resolve_conflict(existing, candidate):
            return True
        overlap = self.modules.memory_retriever.text_similarity(existing.content, candidate.content)
        return overlap < 0.70

    def _merge_confidence(self, existing: MemoryItem, candidate: MemoryItem, similarity: float) -> float:
        importance_signal = min(existing.importance, candidate.importance) / 10.0
        return max(0.0, min(1.0, (0.75 * similarity) + (0.25 * importance_signal)))

    def _find_best_match(self, existing: list[MemoryItem], candidate: MemoryItem) -> tuple[int, float]:
        best_index = -1
        best_similarity = 0.0

        for index, item in enumerate(existing):
            if item.memory_type is not candidate.memory_type:
                continue
            similarity = self.modules.memory_retriever.text_similarity(item.content, candidate.content)
            if similarity > best_similarity:
                best_index = index
                best_similarity = similarity

        return best_index, best_similarity

    def _merge_duplicate(
        self,
        existing: MemoryItem,
        candidate: MemoryItem,
        similarity: float,
        now: datetime,
    ) -> None:
        seen_count = int(existing.metadata.get("seen_count", 1)) + 1
        existing.metadata["seen_count"] = seen_count
        existing.metadata["last_seen_at"] = now.isoformat()
        existing.metadata["dedup_similarity"] = round(similarity, 3)
        existing.importance = min(10.0, max(existing.importance, candidate.importance) + 0.1)
        existing.tags = sorted(set(existing.tags) | set(candidate.tags))

    def _synthesize_semantic_memory(
        self,
        existing: MemoryItem,
        candidate: MemoryItem,
        similarity: float,
        confidence: float,
        now: datetime,
    ) -> MemoryItem:
        merged_content = self._synthesize_summary(existing.content, candidate.content)
        merged_from = list(dict.fromkeys([existing.memory_id, candidate.memory_id]))
        merged_count = int(existing.metadata.get("merged_count", 0)) + 1
        metadata = dict(existing.metadata)
        metadata.update(
            {
                "merged_count": merged_count,
                "last_merged_at": now.isoformat(),
                "merge_similarity": round(similarity, 3),
                "merge_confidence": round(confidence, 3),
                "merge_policy": "semantic_summary_synthesis",
                "merged_from_ids": merged_from,
                "source_fragments": [existing.content, candidate.content],
                "access_count": max(
                    int(existing.metadata.get("access_count", 0)),
                    int(candidate.metadata.get("access_count", 0)),
                ),
            }
        )
        return MemoryItem(
            content=merged_content,
            memory_type=MemoryType.SEMANTIC,
            importance=min(10.0, max(existing.importance, candidate.importance)),
            metadata=metadata,
            tags=sorted(set(existing.tags) | set(candidate.tags) | {"merged", "summary"}),
            created_at=now,
        )

    def _synthesize_summary(self, left: str, right: str) -> str:
        if left == right:
            return left

        prefix = "Summary"
        if ":" in left:
            prefix = left.split(":", 1)[0].strip()

        left_body = left.split(":", 1)[-1].strip()
        right_body = right.split(":", 1)[-1].strip()
        chunks = [part.strip() for part in re.split(r"[|;,\\.]", f"{left_body};{right_body}") if part.strip()]

        unique: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = chunk.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)

        if not unique:
            return left
        summary = " | ".join(unique[:3])
        return f"{prefix}: {summary}"
