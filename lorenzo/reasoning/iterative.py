from __future__ import annotations

from dataclasses import dataclass
import re

from lorenzo.models import InputType, MemoryType, ProcessedInput, ReasoningPlan, RetrievedMemory


@dataclass(slots=True)
class IterativeRefinementResult:
    response: str
    plan: ReasoningPlan
    retrieved_memories: list[RetrievedMemory]
    iterations_used: int
    refinement_triggered: bool
    conflict_detected: bool
    answer_changed: bool
    iteration_gain: float
    trigger_reasons: list[str]
    draft_support_score: float
    final_support_score: float
    factual_refinement_attempted: bool
    factual_refinement_improved: bool
    preference_alignment_attempted: bool
    preference_alignment_improved: bool
    support_completion_attempted: bool
    support_completion_improved: bool
    conflict_fix_attempted: bool
    conflict_fix_succeeded: bool


@dataclass(slots=True)
class RefinementObjectives:
    factual_correction: bool = False
    preference_alignment: bool = False
    conflict_resolution: bool = False
    support_completion: bool = False
    low_confidence: bool = False


class IterativeReasoningEngine:
    def __init__(
        self,
        max_iterations: int = 2,
        low_confidence_threshold: float = 0.38,
        insufficient_similarity_threshold: float = 0.24,
        min_supporting_memories: int = 1,
        support_gain_margin: float = 0.03,
    ) -> None:
        # Keep loop bounded to avoid unbounded reasoning cycles.
        self.max_iterations = max(2, min(3, max_iterations))
        self.low_confidence_threshold = low_confidence_threshold
        self.insufficient_similarity_threshold = insufficient_similarity_threshold
        self.min_supporting_memories = max(1, min_supporting_memories)
        self.support_gain_margin = max(0.0, support_gain_margin)
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

    def refine(
        self,
        *,
        user_input: str,
        processed: ProcessedInput,
        existing_memories,
        initial_retrieved: list[RetrievedMemory],
        draft_plan: ReasoningPlan,
        draft_response: str,
        memory_retriever,
        reasoning_planner,
        language_adapter,
        top_k: int,
    ) -> IterativeRefinementResult:
        current_response = draft_response
        current_plan = draft_plan
        current_retrieved = list(initial_retrieved)

        draft_support = self._support_score(initial_retrieved)
        current_support = draft_support
        refinement_triggered = False
        iterations_used = 1
        trigger_reasons: list[str] = []
        conflict_detected = self._has_conflicting_memories(current_retrieved)
        factual_attempted = False
        factual_improved = False
        preference_attempted = False
        preference_improved = False
        support_attempted = False
        support_improved = False
        conflict_attempted = False
        conflict_fixed = False

        for iteration in range(2, self.max_iterations + 1):
            iterations_used = iteration
            objectives = self._derive_objectives(
                user_input=user_input,
                processed=processed,
                draft_answer=current_response,
                retrieved=current_retrieved,
                current_support=current_support,
            )
            objective_routes = self._objective_routes(objectives)
            if not objective_routes:
                break
            refinement_triggered = True
            trigger_reasons.extend(objective_routes)

            if objectives.factual_correction:
                factual_attempted = True
            if objectives.preference_alignment:
                preference_attempted = True
            if objectives.support_completion:
                support_attempted = True
            if objectives.conflict_resolution:
                conflict_attempted = True

            problematic_claims = self._extract_problematic_claims(
                user_input=user_input,
                draft_answer=current_response,
                retrieved=current_retrieved,
                objectives=objectives,
            )
            unresolved_types = self._prioritized_unresolved_types(objectives)
            requery = self._build_requery(
                user_input=user_input,
                draft_answer=current_response,
                routes=objective_routes,
                problematic_claims=problematic_claims,
                unresolved_types=unresolved_types,
            )
            reretrieved = memory_retriever.retrieve(
                query=requery,
                memories=existing_memories,
                top_k=top_k,
                mode="with_fallback",
            )
            refined_plan = reasoning_planner.plan(processed, reretrieved)
            refined_response = language_adapter.generate(
                user_input=user_input,
                processed=processed,
                retrieved=reretrieved,
                plan=refined_plan,
            )

            refined_support = self._support_score(reretrieved)
            support_gain = refined_support > (current_support + self.support_gain_margin)
            support_not_worse = refined_support >= (current_support - 0.02)
            final_mismatch = self._answer_memory_mismatch(refined_response, reretrieved)
            final_conflict = self._has_conflicting_memories(reretrieved)
            conflict_detected = conflict_detected or objectives.conflict_resolution or final_mismatch

            factual_step_improved = objectives.factual_correction and (
                self._has_fact_support(reretrieved) and not final_mismatch and support_not_worse
            )
            preference_step_improved = objectives.preference_alignment and (
                self._has_preference_support(reretrieved) and support_not_worse
            )
            support_step_improved = objectives.support_completion and (
                (not self._insufficient_supporting_memory(reretrieved)) or support_gain
            )
            conflict_step_fixed = objectives.conflict_resolution and (not final_conflict)

            factual_improved = factual_improved or factual_step_improved
            preference_improved = preference_improved or preference_step_improved
            support_improved = support_improved or support_step_improved
            conflict_fixed = conflict_fixed or conflict_step_fixed

            should_apply_refinement = any(
                [factual_step_improved, preference_step_improved, support_step_improved, conflict_step_fixed]
            )
            if not should_apply_refinement and support_gain and support_not_worse:
                should_apply_refinement = True

            if should_apply_refinement:
                current_response = refined_response
                current_plan = refined_plan
                current_retrieved = reretrieved
                current_support = refined_support

            # v2 currently uses up to 2-pass by default.
            if self.max_iterations <= 2:
                break

        final_trigger_reasons = list(dict.fromkeys(trigger_reasons))
        return IterativeRefinementResult(
            response=current_response,
            plan=current_plan,
            retrieved_memories=current_retrieved,
            iterations_used=iterations_used,
            refinement_triggered=refinement_triggered,
            conflict_detected=conflict_detected,
            answer_changed=current_response.strip() != draft_response.strip(),
            iteration_gain=current_support - draft_support,
            trigger_reasons=final_trigger_reasons,
            draft_support_score=draft_support,
            final_support_score=current_support,
            factual_refinement_attempted=factual_attempted,
            factual_refinement_improved=factual_improved,
            preference_alignment_attempted=preference_attempted,
            preference_alignment_improved=preference_improved,
            support_completion_attempted=support_attempted,
            support_completion_improved=support_improved,
            conflict_fix_attempted=conflict_attempted,
            conflict_fix_succeeded=conflict_fixed,
        )

    def _build_requery(
        self,
        user_input: str,
        draft_answer: str,
        routes: list[str],
        problematic_claims: list[str],
        unresolved_types: list[str],
    ) -> str:
        claims_text = "\n".join(f"- {claim}" for claim in problematic_claims) or "- none"
        routes_text = ", ".join(routes) or "none"
        unresolved_text = ", ".join(unresolved_types) or "general"
        return (
            f"Original query:\n{user_input}\n\n"
            f"Draft answer for refinement:\n{draft_answer}\n\n"
            f"Refinement routes: {routes_text}\n"
            f"Prioritize unresolved memory types: {unresolved_text}\n"
            f"Problematic claims to verify:\n{claims_text}\n\n"
            "Re-check memory consistency, complete missing evidence, and resolve conflicts."
        )

    def _support_score(self, retrieved: list[RetrievedMemory]) -> float:
        if not retrieved:
            return 0.0
        top = retrieved[:2]
        scores: list[float] = []
        for item in top:
            score = (
                (item.similarity_score * 0.65)
                + (item.importance_score * 0.20)
                + (item.recency_score * 0.15)
            )
            scores.append(score)
        return sum(scores) / len(scores)

    def _insufficient_supporting_memory(self, retrieved: list[RetrievedMemory]) -> bool:
        if len(retrieved) < self.min_supporting_memories:
            return True
        return retrieved[0].similarity_score < self.insufficient_similarity_threshold

    def _derive_objectives(
        self,
        *,
        user_input: str,
        processed: ProcessedInput,
        draft_answer: str,
        retrieved: list[RetrievedMemory],
        current_support: float,
    ) -> RefinementObjectives:
        low_confidence = current_support < self.low_confidence_threshold
        insufficient_support = self._insufficient_supporting_memory(retrieved)
        conflict_resolution = self._has_conflicting_memories(retrieved)

        fact_query = self._is_fact_query(user_input=user_input, processed=processed)
        preference_query = self._is_preference_query(user_input=user_input, processed=processed)
        fact_gap = fact_query and (
            insufficient_support
            or (not self._has_fact_support(retrieved))
            or self._answer_memory_mismatch(draft_answer, retrieved)
        )
        preference_mismatch = preference_query and (not self._has_preference_support(retrieved))

        return RefinementObjectives(
            factual_correction=fact_gap,
            preference_alignment=preference_mismatch,
            conflict_resolution=conflict_resolution,
            support_completion=insufficient_support,
            low_confidence=low_confidence,
        )

    def _objective_routes(self, objectives: RefinementObjectives) -> list[str]:
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

    def _prioritized_unresolved_types(self, objectives: RefinementObjectives) -> list[str]:
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

    def _extract_problematic_claims(
        self,
        *,
        user_input: str,
        draft_answer: str,
        retrieved: list[RetrievedMemory],
        objectives: RefinementObjectives,
    ) -> list[str]:
        claims: list[str] = []
        combined = f"{user_input}\n{draft_answer}".lower()
        for key in ["budget", "예산", "deadline", "마감", "price", "가격", "version", "버전"]:
            if key in combined:
                claims.append(f"verify key={key}")

        numbers = re.findall(r"\d+(?:\.\d+)?", draft_answer.lower())
        for number in dict.fromkeys(numbers):
            claims.append(f"verify numeric claim={number}")

        if objectives.conflict_resolution:
            conflict_keys = self._conflicting_keys(retrieved)
            for key in conflict_keys:
                claims.append(f"resolve conflicting key={key}")

        if objectives.preference_alignment:
            claims.append("verify preference style/tone constraints")

        if objectives.support_completion:
            claims.append("collect missing supporting memory evidence")

        return list(dict.fromkeys(claims))

    def _has_conflicting_memories(self, retrieved: list[RetrievedMemory]) -> bool:
        grouped: dict[str, set[str]] = {}
        for item in retrieved:
            memory = item.memory
            if memory.memory_type is not MemoryType.SEMANTIC:
                continue
            if "fact" not in {tag.lower() for tag in memory.tags}:
                continue
            key, value = self._extract_fact_kv(memory.content)
            if not key or not value:
                continue
            grouped.setdefault(key, set()).add(value)
        return any(len(values) > 1 for values in grouped.values())

    def _conflicting_keys(self, retrieved: list[RetrievedMemory]) -> list[str]:
        grouped: dict[str, set[str]] = {}
        for item in retrieved:
            memory = item.memory
            if memory.memory_type is not MemoryType.SEMANTIC:
                continue
            if "fact" not in {tag.lower() for tag in memory.tags}:
                continue
            key, value = self._extract_fact_kv(memory.content)
            if not key or not value:
                continue
            grouped.setdefault(key, set()).add(value)
        return [key for key, values in grouped.items() if len(values) > 1]

    def _extract_fact_kv(self, content: str) -> tuple[str, str]:
        lowered = content.lower()
        structured = re.search(r"key=([a-z_]+)\s*;\s*value=([^;]+)", lowered)
        if structured:
            return structured.group(1).strip(), self._normalize_value(structured.group(2))

        num = re.search(r"(\d+(?:\.\d+)?)", lowered)
        key = ""
        if "budget" in lowered or "예산" in lowered:
            key = "budget"
        elif "deadline" in lowered or "마감" in lowered:
            key = "deadline"
        elif "price" in lowered or "가격" in lowered:
            key = "price"
        elif "version" in lowered or "버전" in lowered:
            key = "version"
        if key and num:
            return key, num.group(1)
        return "", ""

    def _normalize_value(self, raw: str) -> str:
        value = raw.strip().lower()
        numeric = re.search(r"\d+(?:\.\d+)?", value)
        if numeric:
            return numeric.group(0)
        return value

    def _answer_memory_mismatch(self, answer: str, retrieved: list[RetrievedMemory]) -> bool:
        if not answer or not retrieved:
            return False

        answer_lower = answer.lower()
        memory_text = " ".join(item.memory.content.lower() for item in retrieved)
        anchors = [
            ("budget", ["budget", "예산"]),
            ("deadline", ["deadline", "마감"]),
            ("price", ["price", "가격"]),
            ("version", ["version", "버전"]),
        ]
        for _, alias_tokens in anchors:
            if any(token in answer_lower for token in alias_tokens) and not any(
                token in memory_text for token in alias_tokens
            ):
                return True

        return False

    def _is_fact_query(self, *, user_input: str, processed: ProcessedInput) -> bool:
        if InputType.FACT in processed.input_types:
            return True
        lowered = user_input.lower()
        return any(token in lowered for token in self._fact_keywords)

    def _is_preference_query(self, *, user_input: str, processed: ProcessedInput) -> bool:
        if InputType.PREFERENCE in processed.input_types:
            return True
        lowered = user_input.lower()
        return any(token in lowered for token in self._preference_keywords)

    def _has_fact_support(self, retrieved: list[RetrievedMemory]) -> bool:
        return any("fact" in {tag.lower() for tag in item.memory.tags} for item in retrieved[:3])

    def _has_preference_support(self, retrieved: list[RetrievedMemory]) -> bool:
        return any("preference" in {tag.lower() for tag in item.memory.tags} for item in retrieved[:3])
