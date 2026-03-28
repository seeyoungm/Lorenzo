from __future__ import annotations

from dataclasses import dataclass, field

from lorenzo.models import InputType, ProcessedInput, RetrievedMemory


@dataclass(slots=True)
class RefinementObjectives:
    factual_correction: bool = False
    preference_alignment: bool = False
    conflict_resolution: bool = False
    support_completion: bool = False
    low_confidence: bool = False


@dataclass(slots=True)
class RefinementObjectivePolicy:
    low_confidence_threshold: float = 0.38
    insufficient_similarity_threshold: float = 0.24
    min_supporting_memories: int = 1
    # Intent-level claim priority is declarative to avoid spreading hardcoded logic.
    intent_claim_priority: dict[str, list[str]] = field(
        default_factory=lambda: {
            "goal_recall": ["goal", "fact", "preference", "generic"],
            "goal": ["goal", "fact", "preference", "generic"],
            "preference_recall": ["preference", "fact", "goal", "generic"],
            "preference": ["preference", "fact", "goal", "generic"],
            "fact_recall": ["fact", "goal", "preference", "generic"],
            "fact": ["fact", "goal", "preference", "generic"],
            "memory_recall": ["fact", "preference", "goal", "generic"],
            "memory_recall_summary": ["fact", "goal", "preference", "generic"],
            "default": ["fact", "preference", "goal", "generic"],
        }
    )


class RefinementObjectiveRouter:
    def __init__(self, policy: RefinementObjectivePolicy | None = None) -> None:
        self.policy = policy or RefinementObjectivePolicy()
        self._fact_keywords = [
            "fact",
            "사실",
            "예산",
            "budget",
            "deadline",
            "마감",
            "price",
            "가격",
            "version",
            "버전",
        ]
        self._preference_keywords = [
            "preference",
            "prefer",
            "선호",
            "좋아",
            "싫어",
            "style",
            "tone",
            "형식",
            "스타일",
        ]

    def insufficient_supporting_memory(self, retrieved: list[RetrievedMemory]) -> bool:
        if len(retrieved) < self.policy.min_supporting_memories:
            return True
        if not retrieved:
            return True
        return retrieved[0].similarity_score < self.policy.insufficient_similarity_threshold

    def derive(
        self,
        *,
        user_input: str,
        processed: ProcessedInput,
        has_conflict: bool,
        has_fact_support: bool,
        has_preference_support: bool,
        answer_memory_mismatch: bool,
        current_support: float,
        retrieved: list[RetrievedMemory],
    ) -> RefinementObjectives:
        low_confidence = current_support < self.policy.low_confidence_threshold
        insufficient_support = self.insufficient_supporting_memory(retrieved)

        fact_query = self.is_fact_query(user_input=user_input, processed=processed)
        preference_query = self.is_preference_query(user_input=user_input, processed=processed)
        fact_gap = fact_query and (insufficient_support or (not has_fact_support) or answer_memory_mismatch)
        preference_mismatch = preference_query and (not has_preference_support)

        return RefinementObjectives(
            factual_correction=fact_gap,
            preference_alignment=preference_mismatch,
            conflict_resolution=has_conflict or answer_memory_mismatch,
            support_completion=insufficient_support,
            low_confidence=low_confidence,
        )

    def objective_routes(self, objectives: RefinementObjectives) -> list[str]:
        routes: list[str] = []
        if objectives.conflict_resolution:
            routes.append("conflict_triggered_refinement")
        if objectives.factual_correction:
            routes.append("fact_gap_refinement")
        if objectives.preference_alignment:
            routes.append("preference_mismatch_refinement")
        if objectives.support_completion or objectives.low_confidence:
            routes.append("support_completion_refinement")
        return routes

    def prioritized_unresolved_types(self, objectives: RefinementObjectives) -> list[str]:
        types: list[str] = []
        if objectives.factual_correction:
            types.append("fact")
        if objectives.preference_alignment:
            types.append("preference")
        if objectives.conflict_resolution:
            types.append("conflict")
        if objectives.support_completion:
            types.append("support")
        return list(dict.fromkeys(types))

    def claim_priority_for_query(self, user_input: str, processed: ProcessedInput) -> list[str]:
        intent = self.infer_query_intent(user_input=user_input, processed=processed)
        return list(self.policy.intent_claim_priority.get(intent, self.policy.intent_claim_priority["default"]))

    def infer_query_intent(self, *, user_input: str, processed: ProcessedInput) -> str:
        if self.is_fact_query(user_input=user_input, processed=processed):
            return "fact_recall" if "?" in user_input else "fact"
        if self.is_preference_query(user_input=user_input, processed=processed):
            return "preference_recall" if "?" in user_input else "preference"
        if InputType.GOAL_STATEMENT in processed.input_types:
            return "goal"
        lowered = user_input.lower()
        if any(token in lowered for token in ["goal", "목표", "장기", "want"]):
            return "goal_recall" if "?" in lowered else "goal"
        if any(token in lowered for token in ["기억", "remember", "latest", "what was my", "내 "]):
            return "memory_recall"
        return "default"

    def is_fact_query(self, *, user_input: str, processed: ProcessedInput) -> bool:
        if InputType.FACT in processed.input_types:
            return True
        lowered = user_input.lower()
        return any(token in lowered for token in self._fact_keywords)

    def is_preference_query(self, *, user_input: str, processed: ProcessedInput) -> bool:
        if InputType.PREFERENCE in processed.input_types:
            return True
        lowered = user_input.lower()
        return any(token in lowered for token in self._preference_keywords)
