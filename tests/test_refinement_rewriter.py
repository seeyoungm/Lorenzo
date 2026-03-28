from __future__ import annotations

from datetime import datetime, timezone

from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory
from lorenzo.reasoning.refinement_judge import VerificationSummary
from lorenzo.reasoning.refinement_rewriter import ConservativeRefinementRewriter


def _retrieved(content: str) -> RetrievedMemory:
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    return RetrievedMemory(
        memory=MemoryItem(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=["fact", "semantic"],
            created_at=now,
        ),
        total_score=0.8,
        similarity_score=0.8,
        recency_score=0.8,
        importance_score=0.8,
        retrieval_reason="test",
    )


def _summary(unsupported: int, contradicted: int) -> VerificationSummary:
    total = max(1, unsupported + contradicted)
    return VerificationSummary(
        assessments=[],
        support_coverage=0.0,
        unsupported_rate=unsupported / total,
        preference_alignment_score=1.0,
        supported_count=0,
        contradicted_count=contradicted,
        unsupported_count=unsupported,
        avg_evidence_strength=0.0,
    )


def test_rewriter_downgrades_assertive_answer_when_unsupported_remains() -> None:
    rewriter = ConservativeRefinementRewriter()
    rewritten = rewriter.rewrite(
        answer="예산은 500만원이다.",
        summary=_summary(unsupported=1, contradicted=0),
        retrieved=[_retrieved("User fact: key=budget; value=120")],
        reason_flags={"unsupported_claims_remaining": True},
    )

    assert "verification_guarded_response" in rewritten
    assert "단정하기 어렵습니다" in rewritten


def test_rewriter_flags_conflict_when_contradiction_persists() -> None:
    rewriter = ConservativeRefinementRewriter()
    rewritten = rewriter.rewrite(
        answer="마감은 내일이다.",
        summary=_summary(unsupported=0, contradicted=1),
        retrieved=[_retrieved("User fact: key=deadline; value=next week")],
        reason_flags={"contradiction_persisted": True},
    )

    assert "충돌" in rewritten
