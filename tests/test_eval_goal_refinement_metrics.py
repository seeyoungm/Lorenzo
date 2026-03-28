from __future__ import annotations

from pathlib import Path

from lorenzo.config import AppConfig, MemoryConfig
from lorenzo.eval import EvalScenario, evaluate_memory_pipeline


def test_goal_refinement_metrics_are_reported(tmp_path: Path) -> None:
    config = AppConfig(memory=MemoryConfig(path=tmp_path / "seed.jsonl"))
    scenarios = [
        EvalScenario(
            scenario_id="s1",
            category="goal_refinement",
            user_input="장기적으로 메모리 중심 AI 제품을 완성하는 게 목표야",
            expected_retrieval_keywords=[],
            should_store=True,
            expected_store_type="goal",
            expected_conflict_winner_keywords=[],
            stale_conflict_keywords=[],
            session_id="goal-metrics",
            turn=1,
            consistency_group="goal-metrics",
            expected_goal_confidence="strong",
        ),
        EvalScenario(
            scenario_id="s2",
            category="goal_refinement",
            user_input="장기적으로 이런 AI가 되면 좋겠어",
            expected_retrieval_keywords=[],
            should_store=False,
            expected_store_type=None,
            expected_conflict_winner_keywords=[],
            stale_conflict_keywords=[],
            session_id="goal-metrics",
            turn=2,
            consistency_group="goal-metrics",
            expected_goal_confidence="none",
            goal_false_source="wish",
        ),
        EvalScenario(
            scenario_id="s3",
            category="goal_refinement",
            user_input="답변 스타일은 간결한 bullet을 선호해",
            expected_retrieval_keywords=[],
            should_store=True,
            expected_store_type="preference",
            expected_conflict_winner_keywords=[],
            stale_conflict_keywords=[],
            session_id="goal-metrics",
            turn=3,
            consistency_group="goal-metrics",
            expected_goal_confidence="none",
        ),
        EvalScenario(
            scenario_id="s4",
            category="goal_refinement",
            user_input="내 답변 스타일 선호가 뭐야?",
            expected_retrieval_keywords=["간결", "preference"],
            should_store=False,
            expected_store_type=None,
            expected_conflict_winner_keywords=[],
            stale_conflict_keywords=[],
            session_id="goal-metrics",
            turn=4,
            consistency_group="goal-metrics",
            expected_goal_confidence="none",
            goal_intrusion_probe=True,
        ),
    ]

    failure_dump = tmp_path / "refinement_failures.jsonl"
    metrics = evaluate_memory_pipeline(config, scenarios, failure_dump_path=failure_dump)

    assert metrics.goal_precision_strong == 1.0
    assert metrics.goal_recall_strong == 1.0
    assert metrics.false_goal_from_wish_rate == 0.0
    assert 0.0 <= metrics.weak_goal_rate <= 1.0
    assert 0.0 <= metrics.goal_intrusion_rate_in_retrieval_top1 <= 1.0
    assert 0.0 <= metrics.retrieval_hit_rate_top1_strong_only <= 1.0
    assert 0.0 <= metrics.retrieval_hit_rate_top1_with_fallback <= 1.0
    assert 0.0 <= metrics.weak_memory_usage_rate <= 1.0
    assert 0.0 <= metrics.weak_memory_promotion_rate <= 1.0
    assert 0.0 <= metrics.false_positive_reintroduced_rate <= 1.0
    assert 0.0 <= metrics.goal_recall_recovery_rate <= 1.0
    assert 0.0 <= metrics.fallback_trigger_rate <= 1.0
    assert 0.0 <= metrics.fallback_help_rate <= 1.0
    assert 0.0 <= metrics.fallback_no_effect_rate <= 1.0
    assert 0.0 <= metrics.fallback_harm_rate <= 1.0
    assert metrics.avg_rank_change_from_fallback >= -10.0
    assert 0.0 <= metrics.weak_candidate_present_rate <= 1.0
    assert 0.0 <= metrics.weak_candidate_selected_rate <= 1.0
    assert 0.0 <= metrics.weak_coverage_rate <= 1.0
    assert 0.0 <= metrics.weak_coverage_goal_recall <= 1.0
    assert 0.0 <= metrics.weak_coverage_preference_recall <= 1.0
    assert 0.0 <= metrics.weak_coverage_fact_recall <= 1.0
    assert 0.0 <= metrics.weak_override_trigger_rate <= 1.0
    assert 0.0 <= metrics.weak_override_candidate_rate <= 1.0
    assert metrics.weak_override_blocked_by_low_score_count >= 0
    assert metrics.weak_override_blocked_by_similarity_count >= 0
    assert metrics.weak_override_blocked_by_type_alignment_count >= 0
    assert metrics.weak_override_success_count >= 0
    assert 0.0 <= metrics.refinement_improvement_rate <= 1.0
    assert 0.0 <= metrics.conflict_detected_rate <= 1.0
    assert 0.0 <= metrics.answer_change_rate <= 1.0
    assert 0.0 <= metrics.factual_refinement_gain <= 1.0
    assert 0.0 <= metrics.preference_alignment_gain <= 1.0
    assert 0.0 <= metrics.support_completion_gain <= 1.0
    assert 0.0 <= metrics.conflict_fix_rate <= 1.0
    assert 0.0 <= metrics.claim_support_coverage <= 1.0
    assert 0.0 <= metrics.unsupported_claim_rate <= 1.0
    assert 0.0 <= metrics.contradiction_reduction_rate <= 1.0
    assert 0.0 <= metrics.refinement_regression_rate <= 1.0
    assert 0.0 <= metrics.retrieval_improved_but_answer_worsened_rate <= 1.0
    assert 0.0 <= metrics.answer_changed_without_support_improvement_rate <= 1.0
    assert 0.0 <= metrics.unsupported_claim_remaining_rate <= 1.0
    assert failure_dump.exists()
