from __future__ import annotations

from datetime import datetime, timezone

from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory
from lorenzo.reasoning.claim_extractor import Claim
from lorenzo.reasoning.refinement_judge import ClaimAwareRefinementJudge


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


def test_partially_supported_fact_claim_is_supported_with_moderate_strength() -> None:
    judge = ClaimAwareRefinementJudge()
    retrieved = [_retrieved("User fact: key=budget; value=120", ["fact", "semantic"])]
    claim = Claim(claim_type="fact", text="budget=known", key="budget", value="")

    summary = judge.verify_claims([claim], retrieved)
    assessment = summary.assessments[0]

    assert assessment.status == "supported"
    assert assessment.matched_memory_count == 1
    assert assessment.evidence_strength >= 0.6


def test_same_key_conflicting_values_becomes_contradicted() -> None:
    judge = ClaimAwareRefinementJudge()
    retrieved = [
        _retrieved("User fact: key=budget; value=100", ["fact", "semantic"]),
        _retrieved("User fact: key=budget; value=130", ["fact", "semantic"]),
    ]
    claim = Claim(claim_type="fact", text="budget=120", key="budget", value="120")

    summary = judge.verify_claims([claim], retrieved)
    assessment = summary.assessments[0]

    assert assessment.status == "contradicted"
    assert assessment.matched_memory_count == 2


def test_irrelevant_generic_claim_is_not_trivially_supported() -> None:
    judge = ClaimAwareRefinementJudge()
    retrieved = [_retrieved("User event: 오늘 점심 메뉴는 파스타였다", ["event", "episodic"])]
    claim = Claim(claim_type="generic", text="우주 탐사 계획 확정", key="", value="")

    summary = judge.verify_claims([claim], retrieved)

    assert summary.assessments[0].status == "unsupported"
