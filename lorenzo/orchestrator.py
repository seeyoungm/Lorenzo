from __future__ import annotations

from dataclasses import dataclass, field
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
from lorenzo.reasoning import IterativeReasoningEngine, ReasoningPlanner


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
    merge_false_reject_count: int = 0
    merge_false_reject_rate: float = 0.0
    merge_rejected_reason: dict[str, int] = field(default_factory=dict)
    merge_candidate_similarity: list[float] = field(default_factory=list)
    rejected_count_by_type: dict[str, int] = field(default_factory=dict)
    merge_applied: int = 0
    false_merge_attempts: int = 0
    conflict_resolved: int = 0
    conflict_count_by_type: dict[str, int] = field(default_factory=dict)
    weak_memory_candidates: int = 0
    weak_memory_stored: int = 0
    weak_memory_promotions: int = 0
    weak_memory_retrieval_hits: int = 0
    retrieval_turns: int = 0
    refinement_attempts: int = 0
    refinement_triggered_count: int = 0
    refinement_improvement_count: int = 0
    refinement_conflict_detected_count: int = 0
    refinement_answer_changed_count: int = 0
    iteration_gain_total: float = 0.0
    factual_refinement_attempts: int = 0
    factual_refinement_successes: int = 0
    preference_alignment_attempts: int = 0
    preference_alignment_successes: int = 0
    support_completion_attempts: int = 0
    support_completion_successes: int = 0
    conflict_fix_attempts: int = 0
    conflict_fix_successes: int = 0
    claim_support_coverage_total: float = 0.0
    unsupported_claim_rate_total: float = 0.0
    draft_claim_support_coverage_total: float = 0.0
    draft_unsupported_claim_rate_total: float = 0.0
    contradiction_reduction_count: int = 0
    refinement_regression_count: int = 0
    retrieval_improved_but_answer_worsened_count: int = 0
    answer_changed_without_support_gain_count: int = 0
    unsupported_claim_remaining_count: int = 0


@dataclass(slots=True)
class ConflictResolutionResult:
    conflict_strategy: str
    conflict_reason: str
    changed: bool
    conflict_type: str | None = None
    keep_both: bool = False
    requires_confirmation: bool = False


class LorenzoOrchestrator:
    def __init__(
        self,
        modules: PipelineModules,
        top_k: int = 5,
        min_importance_to_store: float = 6.0,
        dedup_similarity_threshold: float = 0.92,
        semantic_merge_similarity_threshold: float = 0.78,
        merge_confidence_threshold: float = 0.82,
        weak_promotion_similarity_threshold: float = 0.60,
        iterative_reasoning_max_iterations: int = 2,
    ) -> None:
        self.modules = modules
        self.top_k = top_k
        self.min_importance_to_store = min_importance_to_store
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.semantic_merge_similarity_threshold = semantic_merge_similarity_threshold
        self.merge_confidence_threshold = merge_confidence_threshold
        self.weak_promotion_similarity_threshold = weak_promotion_similarity_threshold
        self._iterative_reasoning_engine = IterativeReasoningEngine(
            max_iterations=iterative_reasoning_max_iterations
        )
        self._telemetry = UpdateTelemetry()
        self._last_refinement = None

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
                memory_tier_weight=config.memory.memory_tier_weight,
                recency_half_life_hours=config.memory.recency_half_life_hours,
                retrieval_preselect_multiplier=config.memory.retrieval_preselect_multiplier,
                min_similarity_floor=config.memory.min_similarity_floor,
                weak_memory_penalty=config.memory.weak_memory_penalty,
                weak_memory_fallback_penalty=config.memory.weak_memory_fallback_penalty,
                fallback_similarity_threshold=config.memory.fallback_similarity_threshold,
                fallback_score_threshold=config.memory.fallback_score_threshold,
                fallback_min_margin=config.memory.fallback_min_margin,
                weak_override_low_score_threshold=config.memory.weak_override_low_score_threshold,
                weak_override_high_similarity_threshold=config.memory.weak_override_high_similarity_threshold,
                weak_override_type_alignment_threshold=config.memory.weak_override_type_alignment_threshold,
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
            weak_promotion_similarity_threshold=config.memory.weak_promotion_similarity_threshold,
        )

    def run_turn(self, user_input: str) -> TurnResult:
        processed = self.modules.input_processor.process(user_input)

        existing_memories = self.modules.memory_store.list_all()
        retrieved_pass1 = self.modules.memory_retriever.retrieve(
            query=processed.raw_text,
            memories=existing_memories,
            top_k=self.top_k,
            mode="auto",
        )
        draft_plan = self.modules.reasoning_planner.plan(processed, retrieved_pass1)
        draft_response = self.modules.language_adapter.generate(
            user_input=processed.raw_text,
            processed=processed,
            retrieved=retrieved_pass1,
            plan=draft_plan,
        )
        refinement = self._iterative_reasoning_engine.refine(
            user_input=processed.raw_text,
            processed=processed,
            existing_memories=existing_memories,
            initial_retrieved=retrieved_pass1,
            draft_plan=draft_plan,
            draft_response=draft_response,
            memory_retriever=self.modules.memory_retriever,
            reasoning_planner=self.modules.reasoning_planner,
            language_adapter=self.modules.language_adapter,
            top_k=self.top_k,
        )
        self._last_refinement = refinement

        # Keep primary retrieval semantics stable for storage/eval compatibility.
        retrieved = retrieved_pass1
        plan = draft_plan
        if refinement.answer_changed:
            plan = refinement.plan
        response = refinement.response

        self._telemetry.retrieval_turns += 1
        if refinement.refinement_triggered:
            self._telemetry.refinement_attempts += 1
            self._telemetry.refinement_triggered_count += 1
            if refinement.conflict_detected:
                self._telemetry.refinement_conflict_detected_count += 1
            if refinement.answer_changed:
                self._telemetry.refinement_answer_changed_count += 1
            if refinement.refinement_improved:
                self._telemetry.refinement_improvement_count += 1
            self._telemetry.iteration_gain_total += refinement.iteration_gain
            self._telemetry.claim_support_coverage_total += refinement.claim_support_coverage
            self._telemetry.unsupported_claim_rate_total += refinement.unsupported_claim_rate
            self._telemetry.draft_claim_support_coverage_total += (
                refinement.draft_claim_support_coverage
            )
            self._telemetry.draft_unsupported_claim_rate_total += (
                refinement.draft_unsupported_claim_rate
            )
            if refinement.contradiction_reduced:
                self._telemetry.contradiction_reduction_count += 1
            if refinement.refinement_regressed:
                self._telemetry.refinement_regression_count += 1
            if refinement.retrieval_improved_but_answer_worsened:
                self._telemetry.retrieval_improved_but_answer_worsened_count += 1
            if refinement.answer_changed_without_support_improvement:
                self._telemetry.answer_changed_without_support_gain_count += 1
            if refinement.unsupported_claims_remaining:
                self._telemetry.unsupported_claim_remaining_count += 1
        if refinement.factual_refinement_attempted:
            self._telemetry.factual_refinement_attempts += 1
        if refinement.factual_refinement_improved:
            self._telemetry.factual_refinement_successes += 1
        if refinement.preference_alignment_attempted:
            self._telemetry.preference_alignment_attempts += 1
        if refinement.preference_alignment_improved:
            self._telemetry.preference_alignment_successes += 1
        if refinement.support_completion_attempted:
            self._telemetry.support_completion_attempts += 1
        if refinement.support_completion_improved:
            self._telemetry.support_completion_successes += 1
        if refinement.conflict_fix_attempted:
            self._telemetry.conflict_fix_attempts += 1
        if refinement.conflict_fix_succeeded:
            self._telemetry.conflict_fix_successes += 1

        if any(self._memory_tier(item.memory) == "weak" for item in retrieved):
            self._telemetry.weak_memory_retrieval_hits += 1
        if self._apply_retrieval_feedback(existing_memories, retrieved):
            self.modules.memory_store.replace_all(existing_memories)

        self._update_memories(processed)
        return TurnResult(
            user_input=processed.raw_text,
            response=response,
            retrieved_memories=retrieved,
            plan=plan,
        )

    def snapshot_telemetry(self) -> UpdateTelemetry:
        false_reject_rate = (
            self._telemetry.merge_false_reject_count / self._telemetry.merge_rejected_count
            if self._telemetry.merge_rejected_count > 0
            else 0.0
        )
        return UpdateTelemetry(
            turns=self._telemetry.turns,
            candidates_seen=self._telemetry.candidates_seen,
            merge_attempts=self._telemetry.merge_attempts,
            merge_success_count=self._telemetry.merge_success_count,
            merge_rejected_count=self._telemetry.merge_rejected_count,
            merge_false_reject_count=self._telemetry.merge_false_reject_count,
            merge_false_reject_rate=false_reject_rate,
            merge_rejected_reason=dict(self._telemetry.merge_rejected_reason),
            merge_candidate_similarity=list(self._telemetry.merge_candidate_similarity),
            rejected_count_by_type=dict(self._telemetry.rejected_count_by_type),
            merge_applied=self._telemetry.merge_applied,
            false_merge_attempts=self._telemetry.false_merge_attempts,
            conflict_resolved=self._telemetry.conflict_resolved,
            conflict_count_by_type=dict(self._telemetry.conflict_count_by_type),
            weak_memory_candidates=self._telemetry.weak_memory_candidates,
            weak_memory_stored=self._telemetry.weak_memory_stored,
            weak_memory_promotions=self._telemetry.weak_memory_promotions,
            weak_memory_retrieval_hits=self._telemetry.weak_memory_retrieval_hits,
            retrieval_turns=self._telemetry.retrieval_turns,
            refinement_attempts=self._telemetry.refinement_attempts,
            refinement_triggered_count=self._telemetry.refinement_triggered_count,
            refinement_improvement_count=self._telemetry.refinement_improvement_count,
            refinement_conflict_detected_count=self._telemetry.refinement_conflict_detected_count,
            refinement_answer_changed_count=self._telemetry.refinement_answer_changed_count,
            iteration_gain_total=self._telemetry.iteration_gain_total,
            factual_refinement_attempts=self._telemetry.factual_refinement_attempts,
            factual_refinement_successes=self._telemetry.factual_refinement_successes,
            preference_alignment_attempts=self._telemetry.preference_alignment_attempts,
            preference_alignment_successes=self._telemetry.preference_alignment_successes,
            support_completion_attempts=self._telemetry.support_completion_attempts,
            support_completion_successes=self._telemetry.support_completion_successes,
            conflict_fix_attempts=self._telemetry.conflict_fix_attempts,
            conflict_fix_successes=self._telemetry.conflict_fix_successes,
            claim_support_coverage_total=self._telemetry.claim_support_coverage_total,
            unsupported_claim_rate_total=self._telemetry.unsupported_claim_rate_total,
            draft_claim_support_coverage_total=self._telemetry.draft_claim_support_coverage_total,
            draft_unsupported_claim_rate_total=self._telemetry.draft_unsupported_claim_rate_total,
            contradiction_reduction_count=self._telemetry.contradiction_reduction_count,
            refinement_regression_count=self._telemetry.refinement_regression_count,
            retrieval_improved_but_answer_worsened_count=self._telemetry.retrieval_improved_but_answer_worsened_count,
            answer_changed_without_support_gain_count=self._telemetry.answer_changed_without_support_gain_count,
            unsupported_claim_remaining_count=self._telemetry.unsupported_claim_remaining_count,
        )

    def last_refinement_result(self):
        return self._last_refinement

    def _update_memories(self, processed: ProcessedInput) -> None:
        now = datetime.now(timezone.utc)
        existing = self.modules.memory_store.list_all()
        changed = False
        self._telemetry.turns += 1
        candidates = self._build_memory_candidates(processed, now)
        self._telemetry.candidates_seen += len(candidates)

        for candidate in candidates:
            candidate_tier = self._memory_tier(candidate)
            if candidate_tier == "weak":
                self._telemetry.weak_memory_candidates += 1
            is_working_question = candidate.metadata.get("policy") == "question_working_only"
            if not is_working_question and candidate.importance < self.min_importance_to_store:
                continue

            if candidate.memory_type is MemoryType.EPISODIC and "event" in {tag.lower() for tag in candidate.tags}:
                self._attach_event_timestamp(candidate, now)
                if self._mark_event_conflict_if_any(existing, candidate, now):
                    self._telemetry.conflict_resolved += 1
                    self._increment_conflict_type("event")
                existing.append(candidate)
                changed = True
                continue

            match_index, match_similarity = self._find_best_match(existing, candidate)
            if candidate.memory_type is MemoryType.SEMANTIC and match_index >= 0:
                current = existing[match_index]
                current_tier = self._memory_tier(current)
                same_kind = self._semantic_kind(current) == self._semantic_kind(candidate)

                if (
                    candidate_tier == "strong"
                    and current_tier == "weak"
                    and same_kind
                    and match_similarity >= self.weak_promotion_similarity_threshold
                ):
                    existing[match_index] = self._promote_weak_memory(
                        weak_memory=current,
                        strong_candidate=candidate,
                        similarity=match_similarity,
                        now=now,
                    )
                    self._telemetry.weak_memory_promotions += 1
                    changed = True
                    continue

                if (
                    candidate_tier == "weak"
                    and current_tier == "strong"
                    and same_kind
                    and match_similarity >= self.semantic_merge_similarity_threshold
                ):
                    self._record_weak_hint_observed(current, candidate, now, match_similarity)
                    changed = True
                    continue

            if match_index >= 0 and match_similarity >= self.dedup_similarity_threshold:
                self._merge_duplicate(existing[match_index], candidate, match_similarity, now)
                changed = True
                continue

            if candidate.memory_type is MemoryType.SEMANTIC and match_index >= 0:
                resolution = self._resolve_semantic_conflict_by_policy(
                    existing=existing,
                    match_index=match_index,
                    candidate=candidate,
                    similarity=match_similarity,
                    now=now,
                )
                if resolution is not None and resolution.changed:
                    self._telemetry.conflict_resolved += 1
                    if resolution.conflict_type:
                        self._increment_conflict_type(resolution.conflict_type)
                    changed = True
                    continue

            if (
                candidate.memory_type is MemoryType.SEMANTIC
                and match_index >= 0
                and match_similarity >= self.semantic_merge_similarity_threshold
            ):
                self._telemetry.merge_attempts += 1
                self._telemetry.merge_candidate_similarity.append(round(match_similarity, 3))
                confidence = self._merge_confidence(existing[match_index], candidate, match_similarity)
                if confidence < self.merge_confidence_threshold:
                    self._record_merge_rejection(
                        reason="low_confidence",
                        candidate=candidate,
                        similarity=match_similarity,
                        false_reject=(match_similarity >= 0.88),
                    )
                    existing.append(candidate)
                    changed = True
                    continue

                if self._is_false_merge_candidate(existing[match_index], candidate):
                    self._record_merge_rejection(
                        reason="safety_guard",
                        candidate=candidate,
                        similarity=match_similarity,
                        false_reject=False,
                    )
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
            if candidate_tier == "weak":
                self._telemetry.weak_memory_stored += 1
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
                    metadata={
                        "source": "user",
                        "policy": "question_working_only",
                        "memory_tier": "strong",
                    },
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
                    metadata={"rule": "goal_to_semantic", "memory_tier": "strong"},
                    tags=["goal", "semantic", "strong"],
                    created_at=now,
                )
            )
        elif processed.goal_confidence == "weak":
            candidates.append(
                MemoryItem(
                    content=f"Weak goal hint: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(6.6, base_importance - 0.9),
                    metadata={"rule": "weak_goal_hint", "memory_tier": "weak", "trust_level": "hint"},
                    tags=["goal", "semantic", "weak", "weak_hint"],
                    created_at=now,
                )
            )

        if InputType.PREFERENCE in labels:
            candidates.append(
                MemoryItem(
                    content=f"User preference: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.0, base_importance),
                    metadata={"rule": "preference_to_semantic", "memory_tier": "strong"},
                    tags=["preference", "semantic", "strong"],
                    created_at=now,
                )
            )
        elif processed.preference_confidence == "weak":
            candidates.append(
                MemoryItem(
                    content=f"Weak preference hint: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(6.6, base_importance - 0.8),
                    metadata={
                        "rule": "weak_preference_hint",
                        "memory_tier": "weak",
                        "trust_level": "hint",
                        "hint_type": "weak_preference_hint",
                    },
                    tags=[
                        "preference",
                        "semantic",
                        "weak",
                        "weak_hint",
                        "weak_preference_hint",
                    ],
                    created_at=now,
                )
            )

        if InputType.COMMITMENT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User commitment: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.5, base_importance),
                    metadata={"rule": "commitment_to_semantic", "memory_tier": "strong"},
                    tags=["commitment", "semantic", "strong"],
                    created_at=now,
                )
            )

        if InputType.FACT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User fact: {text}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(7.2, base_importance),
                    metadata={"rule": "fact_to_semantic", "memory_tier": "strong"},
                    tags=["fact", "semantic", "strong"],
                    created_at=now,
                )
            )
        elif processed.fact_confidence == "weak":
            weak_fact_key = self._infer_fact_hint_key(text)
            weak_fact_value = self._infer_fact_hint_value(text)
            weak_fact_content = self._build_weak_fact_hint_content(
                source_text=text,
                fact_key=weak_fact_key,
                fact_value=weak_fact_value,
            )
            candidates.append(
                MemoryItem(
                    content=weak_fact_content,
                    memory_type=MemoryType.SEMANTIC,
                    importance=max(6.5, base_importance - 0.9),
                    metadata={
                        "rule": "weak_fact_hint",
                        "memory_tier": "weak",
                        "trust_level": "hint",
                        "hint_type": "weak_fact_hint",
                        "fact_key": weak_fact_key,
                        "fact_value": weak_fact_value,
                    },
                    tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
                    created_at=now,
                )
            )

        if InputType.EVENT in labels:
            candidates.append(
                MemoryItem(
                    content=f"User event: {text}",
                    memory_type=MemoryType.EPISODIC,
                    importance=max(6.5, base_importance),
                    metadata={"rule": "event_to_episodic", "memory_tier": "strong"},
                    tags=["event", "episodic", "strong"],
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

    def _infer_fact_hint_key(self, text: str) -> str:
        lowered = text.lower()
        key_aliases = (
            ("budget", ("budget", "예산", "비용", "cost")),
            ("deadline", ("deadline", "마감", "기한", "due")),
            ("price", ("price", "가격", "요금")),
            ("version", ("version", "버전", "릴리스")),
            ("rate", ("rate", "비율", "percent", "%")),
        )
        for canonical_key, aliases in key_aliases:
            if any(alias in lowered for alias in aliases):
                return canonical_key
        return ""

    def _infer_fact_hint_value(self, text: str) -> str:
        numeric = re.search(r"\d+(?:[.,]\d+)?(?::\d{2})?", text)
        if numeric:
            return numeric.group(0).replace(",", "")

        korean_tail = re.search(r"(?:은|는|이|가)\s*(.+)$", text.strip())
        if korean_tail:
            return korean_tail.group(1).strip().lower()

        english_tail = re.search(r"\bis\s+(.+)$", text.strip(), flags=re.IGNORECASE)
        if english_tail:
            return english_tail.group(1).strip().lower()

        return ""

    def _build_weak_fact_hint_content(self, source_text: str, fact_key: str, fact_value: str) -> str:
        fields: list[str] = []
        if fact_key:
            fields.append(f"key={fact_key}")
        if fact_value:
            fields.append(f"value={fact_value}")
        if not fields:
            return f"Weak fact hint: source={source_text}"
        return f"Weak fact hint: {'; '.join(fields)}; source={source_text}"

    def _resolve_semantic_conflict_by_policy(
        self,
        existing: list[MemoryItem],
        match_index: int,
        candidate: MemoryItem,
        similarity: float,
        now: datetime,
    ) -> ConflictResolutionResult | None:
        current = existing[match_index]
        if self._memory_tier(current) != "strong" or self._memory_tier(candidate) != "strong":
            return None
        existing_kind = self._semantic_kind(current)
        candidate_kind = self._semantic_kind(candidate)
        if existing_kind != candidate_kind:
            return None

        if candidate_kind == "goal":
            if not self._is_semantic_conflict_candidate(current, candidate, similarity):
                return None
            self._apply_latest_wins_with_history(
                current,
                candidate,
                now,
                history_key="goal_history",
                history_status="archived",
            )
            self._annotate_conflict_resolution(
                current,
                conflict_strategy="latest_wins",
                conflict_reason="goal_update_latest_wins",
                now=now,
            )
            return ConflictResolutionResult(
                conflict_strategy="latest_wins",
                conflict_reason="goal_update_latest_wins",
                changed=True,
                conflict_type="goal",
            )

        if candidate_kind == "preference":
            if not self._is_semantic_conflict_candidate(current, candidate, similarity):
                return None
            self._apply_latest_wins_with_history(
                current,
                candidate,
                now,
                history_key="preference_history",
                history_status="inactive",
            )
            self._annotate_conflict_resolution(
                current,
                conflict_strategy="latest_wins",
                conflict_reason="preference_update_latest_wins",
                now=now,
            )
            return ConflictResolutionResult(
                conflict_strategy="latest_wins",
                conflict_reason="preference_update_latest_wins",
                changed=True,
                conflict_type="preference",
            )

        if candidate_kind == "fact":
            if not self._is_fact_conflict(current, candidate):
                return None
            self._mark_fact_conflict_keep_both(current, candidate, now)
            existing.append(candidate)
            return ConflictResolutionResult(
                conflict_strategy="keep_both_mark_conflict",
                conflict_reason="fact_value_conflict",
                changed=True,
                conflict_type="fact",
                keep_both=True,
            )

        if candidate_kind == "commitment":
            if not self._is_semantic_conflict_candidate(current, candidate, similarity):
                return None
            if self._has_explicit_confirmation_signal(candidate.content):
                self._apply_latest_wins_with_history(
                    current,
                    candidate,
                    now,
                    history_key="commitment_history",
                    history_status="superseded_confirmed",
                )
                self._annotate_conflict_resolution(
                    current,
                    conflict_strategy="latest_wins_confirmed",
                    conflict_reason="commitment_update_with_confirmation",
                    now=now,
                )
                return ConflictResolutionResult(
                    conflict_strategy="latest_wins_confirmed",
                    conflict_reason="commitment_update_with_confirmation",
                    changed=True,
                    conflict_type="commitment",
                )

            self._mark_commitment_conflict_pending(current, candidate, now)
            existing.append(candidate)
            return ConflictResolutionResult(
                conflict_strategy="require_explicit_confirmation",
                conflict_reason="commitment_conflict_pending_confirmation",
                changed=True,
                conflict_type="commitment",
                keep_both=True,
                requires_confirmation=True,
            )

        return None

    def _semantic_kind(self, memory: MemoryItem) -> str:
        tags = {tag.lower() for tag in memory.tags}
        if "goal" in tags or memory.content.startswith("User goal:"):
            return "goal"
        if "preference" in tags or memory.content.startswith("User preference:"):
            return "preference"
        if "fact" in tags or memory.content.startswith("User fact:"):
            return "fact"
        if "commitment" in tags or memory.content.startswith("User commitment:"):
            return "commitment"
        return "unknown"

    def _memory_tier(self, memory: MemoryItem) -> str:
        tier = str(memory.metadata.get("memory_tier", "strong")).lower().strip()
        if tier not in {"strong", "weak"}:
            return "strong"
        return tier

    def _promote_weak_memory(
        self,
        weak_memory: MemoryItem,
        strong_candidate: MemoryItem,
        similarity: float,
        now: datetime,
    ) -> MemoryItem:
        metadata = dict(weak_memory.metadata)
        history = list(metadata.get("weak_history", []))
        history.append(
            {
                "content": weak_memory.content,
                "promoted_at": now.isoformat(),
                "similarity": round(similarity, 3),
            }
        )
        metadata.update(strong_candidate.metadata)
        metadata["memory_tier"] = "strong"
        metadata["promoted_from_weak"] = True
        metadata["promotion_count"] = int(metadata.get("promotion_count", 0)) + 1
        metadata["weak_history"] = history[-20:]
        metadata["promotion_similarity"] = round(similarity, 3)
        metadata["promoted_at"] = now.isoformat()
        return MemoryItem(
            content=strong_candidate.content,
            memory_type=strong_candidate.memory_type,
            importance=max(weak_memory.importance, strong_candidate.importance),
            metadata=metadata,
            tags=sorted(
                {
                    tag
                    for tag in (list(weak_memory.tags) + list(strong_candidate.tags) + ["strong"])
                    if tag.lower() not in {"weak", "weak_hint"}
                }
            ),
            created_at=now,
        )

    def _record_weak_hint_observed(
        self,
        strong_memory: MemoryItem,
        weak_candidate: MemoryItem,
        now: datetime,
        similarity: float,
    ) -> None:
        hints = list(strong_memory.metadata.get("weak_hints_ignored", []))
        hints.append(
            {
                "at": now.isoformat(),
                "content": weak_candidate.content,
                "similarity": round(similarity, 3),
            }
        )
        strong_memory.metadata["weak_hints_ignored"] = hints[-20:]
        strong_memory.metadata["weak_hint_seen_count"] = int(
            strong_memory.metadata.get("weak_hint_seen_count", 0)
        ) + 1

    def _is_semantic_conflict_candidate(
        self,
        existing: MemoryItem,
        candidate: MemoryItem,
        similarity: float,
    ) -> bool:
        return similarity >= 0.72 and existing.content.strip().lower() != candidate.content.strip().lower()

    def _apply_latest_wins_with_history(
        self,
        existing: MemoryItem,
        candidate: MemoryItem,
        now: datetime,
        history_key: str,
        history_status: str,
    ) -> None:
        previous_content = existing.content
        previous_importance = existing.importance
        history = list(existing.metadata.get(history_key, []))
        history.append(
            {
                "content": previous_content,
                "importance": previous_importance,
                "status": history_status,
                "replaced_at": now.isoformat(),
            }
        )
        existing.content = candidate.content
        existing.importance = min(10.0, max(existing.importance, candidate.importance))
        existing.metadata[history_key] = history[-20:]
        existing.metadata["memory_tier"] = "strong"
        existing.tags = sorted(
            {
                tag
                for tag in (list(existing.tags) + list(candidate.tags) + ["conflict_resolved", "strong"])
                if tag.lower() not in {"weak", "weak_hint"}
            }
        )

    def _annotate_conflict_resolution(
        self,
        memory: MemoryItem,
        conflict_strategy: str,
        conflict_reason: str,
        now: datetime,
    ) -> None:
        memory.metadata["conflict_strategy"] = conflict_strategy
        memory.metadata["conflict_reason"] = conflict_reason
        memory.metadata["last_conflict_at"] = now.isoformat()

    def _is_fact_conflict(self, existing: MemoryItem, candidate: MemoryItem) -> bool:
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

    def _mark_fact_conflict_keep_both(self, existing: MemoryItem, candidate: MemoryItem, now: datetime) -> None:
        existing_history = list(existing.metadata.get("fact_conflicts", []))
        entry = {
            "at": now.isoformat(),
            "existing": existing.content,
            "incoming": candidate.content,
        }
        existing_history.append(entry)
        existing.metadata["fact_conflicts"] = existing_history[-20:]
        existing.metadata["conflict_strategy"] = "keep_both_mark_conflict"
        existing.metadata["conflict_reason"] = "fact_value_conflict"
        existing.metadata["memory_tier"] = "strong"
        existing.tags = sorted(set(existing.tags) | {"conflict"})

        candidate.metadata["conflict_strategy"] = "keep_both_mark_conflict"
        candidate.metadata["conflict_reason"] = "fact_value_conflict"
        candidate.metadata["conflict_with"] = existing.memory_id
        candidate.metadata["conflict_recorded_at"] = now.isoformat()
        candidate.metadata["memory_tier"] = "strong"
        candidate.tags = sorted(set(candidate.tags) | {"conflict"})

    def _mark_commitment_conflict_pending(
        self, existing: MemoryItem, candidate: MemoryItem, now: datetime
    ) -> None:
        pending = list(existing.metadata.get("commitment_pending_conflicts", []))
        pending.append(
            {
                "at": now.isoformat(),
                "existing": existing.content,
                "incoming": candidate.content,
                "status": "awaiting_confirmation",
            }
        )
        existing.metadata["commitment_pending_conflicts"] = pending[-20:]
        existing.metadata["conflict_strategy"] = "require_explicit_confirmation"
        existing.metadata["conflict_reason"] = "commitment_conflict_pending_confirmation"
        existing.metadata["memory_tier"] = "strong"
        existing.tags = sorted(set(existing.tags) | {"conflict_pending"})

        candidate.metadata["conflict_strategy"] = "require_explicit_confirmation"
        candidate.metadata["conflict_reason"] = "commitment_conflict_pending_confirmation"
        candidate.metadata["needs_confirmation"] = True
        candidate.metadata["conflict_with"] = existing.memory_id
        candidate.metadata["memory_tier"] = "strong"
        candidate.tags = sorted(set(candidate.tags) | {"conflict_pending"})

    def _has_explicit_confirmation_signal(self, content: str) -> bool:
        lowered = content.lower()
        signals = [
            "confirm",
            "confirmed",
            "override",
            "overwrite",
            "덮어써",
            "덮어쓰기",
            "확정",
            "확인",
            "변경 확정",
        ]
        return any(token in lowered for token in signals)

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

    def _attach_event_timestamp(self, candidate: MemoryItem, now: datetime) -> None:
        event_ts = self._extract_event_timestamp(candidate.content) or now.isoformat()
        candidate.metadata["event_timestamp"] = event_ts
        candidate.metadata["conflict_strategy"] = "preserve_by_timestamp"
        candidate.metadata["conflict_reason"] = "event_timeline_append"

    def _mark_event_conflict_if_any(
        self,
        existing: list[MemoryItem],
        candidate: MemoryItem,
        now: datetime,
    ) -> bool:
        candidate_key, candidate_value = self._extract_event_kv(candidate.content)
        if not candidate_key or not candidate_value:
            return False

        conflicted = False
        for item in existing:
            if item.memory_type is not MemoryType.EPISODIC:
                continue
            if "event" not in {tag.lower() for tag in item.tags}:
                continue
            existing_key, existing_value = self._extract_event_kv(item.content)
            if existing_key != candidate_key or not existing_value:
                continue
            if existing_value == candidate_value:
                continue

            history = list(item.metadata.get("event_conflicts", []))
            history.append(
                {
                    "at": now.isoformat(),
                    "previous": item.content,
                    "incoming": candidate.content,
                    "strategy": "preserve_by_timestamp",
                }
            )
            item.metadata["event_conflicts"] = history[-20:]
            item.metadata["conflict_strategy"] = "preserve_by_timestamp"
            item.metadata["conflict_reason"] = "event_time_conflict_preserved"
            item.tags = sorted(set(item.tags) | {"conflict"})
            conflicted = True

        if conflicted:
            candidate.metadata["conflict_strategy"] = "preserve_by_timestamp"
            candidate.metadata["conflict_reason"] = "event_time_conflict_preserved"
            candidate.tags = sorted(set(candidate.tags) | {"conflict"})
        return conflicted

    def _extract_event_kv(self, content: str) -> tuple[str, str]:
        normalized = content.replace("User event:", "").strip().lower()
        kr = re.search(r"(회의|미팅).{0,24}?(\d{1,2}\s*시)", normalized)
        if kr:
            return "meeting_time", kr.group(2).replace(" ", "")
        en = re.search(r"(meeting|appointment).{0,32}?(\d{1,2}:\d{2}\s*(am|pm)?)", normalized)
        if en:
            return "meeting_time", en.group(2).replace(" ", "")
        return "", ""

    def _extract_event_timestamp(self, content: str) -> str | None:
        normalized = content.lower()
        with_date = re.search(r"(\d{4}-\d{2}-\d{2}[ t]\d{1,2}:\d{2})", normalized)
        if with_date:
            return with_date.group(1)
        relative = re.search(r"(오늘|내일|모레).{0,16}?(\d{1,2}\s*시)", normalized)
        if relative:
            return f"{relative.group(1)}-{relative.group(2).replace(' ', '')}"
        return None

    def _is_false_merge_candidate(self, existing: MemoryItem, candidate: MemoryItem) -> bool:
        if self._semantic_kind(existing) in {"fact", "commitment"}:
            return True
        overlap = self.modules.memory_retriever.text_similarity(existing.content, candidate.content)
        return overlap < 0.70

    def _record_merge_rejection(
        self,
        reason: str,
        candidate: MemoryItem,
        similarity: float,
        false_reject: bool,
    ) -> None:
        self._telemetry.merge_rejected_count += 1
        self._telemetry.merge_rejected_reason[reason] = (
            self._telemetry.merge_rejected_reason.get(reason, 0) + 1
        )
        kind = (
            self._semantic_kind(candidate)
            if candidate.memory_type is MemoryType.SEMANTIC
            else candidate.memory_type.value
        )
        self._telemetry.rejected_count_by_type[kind] = self._telemetry.rejected_count_by_type.get(kind, 0) + 1
        candidate.metadata["merge_rejected_reason"] = reason
        candidate.metadata["merge_candidate_similarity"] = round(similarity, 3)
        if false_reject:
            self._telemetry.merge_false_reject_count += 1

    def _increment_conflict_type(self, conflict_type: str) -> None:
        self._telemetry.conflict_count_by_type[conflict_type] = (
            self._telemetry.conflict_count_by_type.get(conflict_type, 0) + 1
        )

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
        existing_tier = self._memory_tier(existing)
        candidate_tier = self._memory_tier(candidate)
        merged_tier = "strong" if "strong" in {existing_tier, candidate_tier} else "weak"
        seen_count = int(existing.metadata.get("seen_count", 1)) + 1
        existing.metadata["seen_count"] = seen_count
        existing.metadata["last_seen_at"] = now.isoformat()
        existing.metadata["dedup_similarity"] = round(similarity, 3)
        existing.metadata["memory_tier"] = merged_tier
        existing.importance = min(10.0, max(existing.importance, candidate.importance) + 0.1)
        merged_tags = sorted(set(existing.tags) | set(candidate.tags))
        if merged_tier == "strong":
            merged_tags = [
                tag for tag in merged_tags if tag.lower() not in {"weak", "weak_hint"}
            ] + ["strong"]
        existing.tags = sorted(set(merged_tags))

    def _synthesize_semantic_memory(
        self,
        existing: MemoryItem,
        candidate: MemoryItem,
        similarity: float,
        confidence: float,
        now: datetime,
    ) -> MemoryItem:
        existing_tier = self._memory_tier(existing)
        candidate_tier = self._memory_tier(candidate)
        merged_tier = "strong" if "strong" in {existing_tier, candidate_tier} else "weak"
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
                "memory_tier": merged_tier,
            }
        )
        merged_tags = sorted(set(existing.tags) | set(candidate.tags) | {"merged", "summary"})
        if merged_tier == "strong":
            merged_tags = [tag for tag in merged_tags if tag.lower() not in {"weak", "weak_hint"}]
            merged_tags = sorted(set(merged_tags) | {"strong"})
        return MemoryItem(
            content=merged_content,
            memory_type=MemoryType.SEMANTIC,
            importance=min(10.0, max(existing.importance, candidate.importance)),
            metadata=metadata,
            tags=merged_tags,
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
