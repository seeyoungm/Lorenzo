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

    metrics = evaluate_memory_pipeline(config, scenarios)

    assert metrics.goal_precision_strong == 1.0
    assert metrics.goal_recall_strong == 1.0
    assert metrics.false_goal_from_wish_rate == 0.0
    assert 0.0 <= metrics.weak_goal_rate <= 1.0
    assert 0.0 <= metrics.goal_intrusion_rate_in_retrieval_top1 <= 1.0
