from __future__ import annotations

from dataclasses import dataclass
import re

from lorenzo.models import MemoryType, ProcessedInput, ReasoningPlan, RetrievedMemory


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

        for iteration in range(2, self.max_iterations + 1):
            iterations_used = iteration
            requery = self._build_requery(user_input=user_input, draft_answer=current_response)
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
            low_confidence = current_support < self.low_confidence_threshold
            insufficient_support = self._insufficient_supporting_memory(current_retrieved)
            conflict_now = self._has_conflicting_memories(current_retrieved) or self._has_conflicting_memories(
                reretrieved
            )
            mismatch_now = self._answer_memory_mismatch(refined_response, reretrieved)
            conflict_detected = conflict_detected or conflict_now or mismatch_now

            iteration_reasons: list[str] = []
            if low_confidence:
                iteration_reasons.append("low_confidence")
            if insufficient_support:
                iteration_reasons.append("insufficient_supporting_memory")
            if conflict_now:
                iteration_reasons.append("conflicting_memory")
            if mismatch_now:
                iteration_reasons.append("answer_memory_mismatch")
            if refined_support > (current_support + self.support_gain_margin):
                iteration_reasons.append("support_gain")

            if iteration_reasons:
                refinement_triggered = True
                trigger_reasons.extend(iteration_reasons)

            support_gain = refined_support > (current_support + self.support_gain_margin)
            support_not_worse = refined_support >= (current_support - 0.02)
            should_apply_refinement = False
            if support_gain:
                should_apply_refinement = True
            elif (low_confidence or insufficient_support) and support_not_worse:
                should_apply_refinement = True

            if iteration_reasons:
                if not should_apply_refinement:
                    continue
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
        )

    def _build_requery(self, user_input: str, draft_answer: str) -> str:
        return (
            f"Original query:\n{user_input}\n\n"
            f"Draft answer for refinement:\n{draft_answer}\n\n"
            "Re-check memory consistency and missing details."
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
