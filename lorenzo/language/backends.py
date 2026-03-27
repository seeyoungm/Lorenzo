from __future__ import annotations

from typing import Protocol

from lorenzo.models import ProcessedInput, ReasoningPlan, RetrievedMemory


class LanguageBackend(Protocol):
    def generate(
        self,
        user_input: str,
        processed: ProcessedInput,
        retrieved: list[RetrievedMemory],
        plan: ReasoningPlan,
    ) -> str:
        ...


class RuleBasedBackend:
    """Simple backend for local/offline demos."""

    def generate(
        self,
        user_input: str,
        processed: ProcessedInput,
        retrieved: list[RetrievedMemory],
        plan: ReasoningPlan,
    ) -> str:
        memory_summary = ""
        if retrieved:
            top = retrieved[:2]
            bullets = "\n".join(
                f"- {item.memory.content} (reason: {item.retrieval_reason})"
                for item in top
            )
            memory_summary = f"참고 기억:\n{bullets}\n\n"
        else:
            memory_summary = "참고 기억: 없음\n\n"

        next_actions = "\n".join(f"- {step}" for step in plan.steps)
        return (
            f"[전략] {plan.strategy}\n"
            f"[근거] {plan.rationale}\n\n"
            f"{memory_summary}"
            f"입력 해석: {processed.input_type.value}\n"
            f"권장 응답:\n"
            f"{next_actions}\n\n"
            f"요약 답변: '{user_input}' 요청을 메모리 기반으로 처리했습니다."
        )


class EchoBackend:
    def generate(
        self,
        user_input: str,
        processed: ProcessedInput,
        retrieved: list[RetrievedMemory],
        plan: ReasoningPlan,
    ) -> str:
        return (
            f"Echo<{processed.input_type.value}>: {user_input} "
            f"(strategy={plan.strategy}, retrieved={len(retrieved)})"
        )
