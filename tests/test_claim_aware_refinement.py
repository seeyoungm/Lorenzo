from __future__ import annotations

from datetime import datetime, timezone

from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory
from lorenzo.reasoning.claim_extractor import Claim
from lorenzo.reasoning.refinement_judge import (
    ClaimAssessment,
    ClaimAwareRefinementJudge,
    VerificationSummary,
)
from lorenzo.reasoning.refinement_objectives import RefinementObjectiveRouter
from lorenzo.reasoning.requery_builder import RequeryBuilder


def _retrieved(content: str, tags: list[str]) -> RetrievedMemory:
    now = datetime(2026, 3, 28, tzinfo=timezone.utc)
    return RetrievedMemory(
        memory=MemoryItem(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=tags,
            created_at=now,
        ),
        total_score=0.8,
        similarity_score=0.8,
        recency_score=0.8,
        importance_score=0.8,
        retrieval_reason="test",
    )


def _summary(
    support_coverage: float,
    unsupported_rate: float,
    contradicted_count: int,
    unsupported_count: int,
    preference_alignment_score: float = 1.0,
    avg_evidence_strength: float = 0.5,
) -> VerificationSummary:
    return VerificationSummary(
        assessments=[],
        support_coverage=support_coverage,
        unsupported_rate=unsupported_rate,
        preference_alignment_score=preference_alignment_score,
        supported_count=0,
        contradicted_count=contradicted_count,
        unsupported_count=unsupported_count,
        avg_evidence_strength=avg_evidence_strength,
    )


def test_claim_aware_verifier_classifies_fact_supported_contradicted_unsupported() -> None:
    judge = ClaimAwareRefinementJudge()
    retrieved = [
        _retrieved("User fact: key=budget; value=120", ["fact", "semantic"]),
        _retrieved("User fact: key=budget; value=140", ["fact", "semantic"]),
    ]
    claims = [
        Claim(claim_type="fact", text="budget=120", key="budget", value="120"),
        Claim(claim_type="fact", text="budget=200", key="budget", value="200"),
        Claim(claim_type="fact", text="deadline=unknown", key="deadline", value=""),
    ]

    summary = judge.verify_claims(claims, retrieved)
    statuses = [item.status for item in summary.assessments]

    assert statuses == ["supported", "contradicted", "unsupported"]


def test_refinement_judge_rejects_retrieval_improved_but_answer_worsened() -> None:
    judge = ClaimAwareRefinementJudge()
    draft = _summary(support_coverage=0.8, unsupported_rate=0.1, contradicted_count=0, unsupported_count=1)
    refined = _summary(support_coverage=0.6, unsupported_rate=0.3, contradicted_count=1, unsupported_count=2)

    decision = judge.judge_refinement(
        draft_summary=draft,
        refined_summary=refined,
        draft_support_score=0.30,
        refined_support_score=0.55,
        support_gain_margin=0.03,
    )

    assert decision.apply_refinement is False
    assert decision.regression is True
    assert decision.retrieval_improved_but_answer_worsened is True


def test_requery_builder_prioritizes_unresolved_claims() -> None:
    builder = RequeryBuilder()
    assessments = [
        ClaimAssessment(
            claim=Claim(claim_type="fact", text="budget=120", key="budget", value="120"),
            status="unsupported",
            evidence_memory_ids=[],
        ),
        ClaimAssessment(
            claim=Claim(claim_type="preference", text="prefer concise", key="style", value="concise"),
            status="supported",
            evidence_memory_ids=["1"],
        ),
    ]

    query = builder.build(
        user_input="예산이 얼마였지?",
        draft_answer="예산은 120이야",
        routes=["fact_gap_refinement"],
        unresolved_types=["fact"],
        claim_priority=["fact", "preference", "goal", "generic"],
        claim_assessments=assessments,
    )

    assert "fact_gap_refinement" in query
    assert "verify key=budget; value=120" in query


def test_objective_router_has_intent_claim_priority_table() -> None:
    router = RefinementObjectiveRouter()
    priorities = router.policy.intent_claim_priority

    assert "goal_recall" in priorities
    assert priorities["goal_recall"][0] == "goal"
