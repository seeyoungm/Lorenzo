from __future__ import annotations

from lorenzo.models import RetrievedMemory
from lorenzo.reasoning.refinement_judge import VerificationSummary


class ConservativeRefinementRewriter:
    """Rewrites risky answers into evidence-bounded responses."""

    def rewrite(
        self,
        *,
        answer: str,
        summary: VerificationSummary,
        retrieved: list[RetrievedMemory],
        reason_flags: dict[str, bool] | None = None,
    ) -> str:
        flags = reason_flags or {}
        lines: list[str] = [
            "[전략] verification_guarded_response",
            "[근거] 검증되지 않은 주장/충돌 가능성을 줄이기 위해 단정 표현을 낮춘다.",
            "",
        ]

        if summary.unsupported_count > 0:
            lines.append("현재 기억 근거만으로는 일부 항목을 단정하기 어렵습니다.")
        if summary.contradicted_count > 0 or flags.get("contradiction_persisted", False):
            lines.append("관련 기억 간 충돌이 있어 최신/확정 정보 확인이 필요합니다.")
        if flags.get("answer_memory_mismatch", False):
            lines.append("응답-기억 정합성이 낮아 보수적으로 답변합니다.")

        lines.append("")
        lines.append("확인 가능한 기억:")
        for item in retrieved[:2]:
            lines.append(f"- {item.memory.content[:180]}")
        if not retrieved:
            lines.append("- (검증 가능한 기억 없음)")

        lines.append("")
        lines.append("확실하지 않은 부분은 추정하지 않고 확인 요청으로 남깁니다.")
        return "\n".join(lines).strip()
