from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning.planner import ReasoningPlanner


def _build(path: Path, merge_confidence_threshold: float = 0.68) -> LorenzoOrchestrator:
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=JsonlMemoryStore(path),
        memory_retriever=MemoryRetriever(),
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(EchoBackend()),
    )
    return LorenzoOrchestrator(
        modules=modules,
        top_k=5,
        min_importance_to_store=6.5,
        dedup_similarity_threshold=0.99,
        semantic_merge_similarity_threshold=0.60,
        merge_confidence_threshold=merge_confidence_threshold,
    )


def test_goal_conflict_uses_latest_wins_with_history(tmp_path: Path) -> None:
    orch = _build(tmp_path / "goal.jsonl")

    orch.run_turn("나는 장기 기억 AI를 만들고 싶어")
    orch.run_turn("나는 장기 기억 AI를 제품 수준으로 만들고 싶어")

    goals = [
        item
        for item in orch.modules.memory_store.list_all()
        if item.content.startswith("User goal:")
    ]
    assert len(goals) == 1
    goal = goals[0]
    assert "제품 수준" in goal.content
    assert goal.metadata.get("conflict_strategy") == "latest_wins"
    assert goal.metadata.get("conflict_reason") == "goal_update_latest_wins"
    history = goal.metadata.get("goal_history", [])
    assert history and history[0].get("status") == "archived"


def test_preference_conflict_marks_previous_inactive(tmp_path: Path) -> None:
    orch = _build(tmp_path / "preference.jsonl")

    orch.run_turn("나는 코드 예시는 짧게 주는 걸 선호해")
    orch.run_turn("나는 예시는 자세하게 주는 걸 선호해")

    prefs = [
        item
        for item in orch.modules.memory_store.list_all()
        if item.content.startswith("User preference:")
    ]
    assert len(prefs) == 1
    pref = prefs[0]
    assert "자세하게" in pref.content
    assert pref.metadata.get("conflict_strategy") == "latest_wins"
    assert pref.metadata.get("conflict_reason") == "preference_update_latest_wins"
    history = pref.metadata.get("preference_history", [])
    assert history and history[0].get("status") == "inactive"


def test_commitment_conflict_requires_confirmation(tmp_path: Path) -> None:
    orch = _build(tmp_path / "commitment.jsonl")

    orch.run_turn("나는 이번 주 안에 문서를 제출할게")
    orch.run_turn("나는 이번 주 안에 보고서를 제출할게")

    commitments = [
        item
        for item in orch.modules.memory_store.list_all()
        if item.content.startswith("User commitment:")
    ]
    assert len(commitments) == 2
    assert any(item.metadata.get("needs_confirmation") is True for item in commitments)
    assert any(
        item.metadata.get("conflict_strategy") == "require_explicit_confirmation" for item in commitments
    )
    assert any(
        item.metadata.get("conflict_reason") == "commitment_conflict_pending_confirmation"
        for item in commitments
    )


def test_commitment_conflict_overwrites_when_explicitly_confirmed(tmp_path: Path) -> None:
    orch = _build(tmp_path / "commitment-confirmed.jsonl")

    orch.run_turn("나는 이번 주 안에 문서를 제출할게")
    orch.run_turn("변경 확정: 나는 이번 주 안에 테스트 계획을 제출할게")

    commitments = [
        item
        for item in orch.modules.memory_store.list_all()
        if item.content.startswith("User commitment:")
    ]
    assert len(commitments) == 1
    current = commitments[0]
    assert "테스트 계획" in current.content
    assert current.metadata.get("conflict_strategy") == "latest_wins_confirmed"
    assert current.metadata.get("conflict_reason") == "commitment_update_with_confirmation"
    history = current.metadata.get("commitment_history", [])
    assert history and history[0].get("status") == "superseded_confirmed"


def test_merge_rejection_observability_tracks_reason_and_type(tmp_path: Path) -> None:
    orch = _build(tmp_path / "merge-observe.jsonl", merge_confidence_threshold=0.99)

    orch.run_turn("중요: 프로젝트 예산은 80만원이다")
    orch.run_turn("중요: 현재 프로젝트 예산은 80만원이다")

    telemetry = orch.snapshot_telemetry()
    assert telemetry.merge_rejected_count >= 1
    assert telemetry.merge_rejected_reason.get("low_confidence", 0) >= 1
    assert telemetry.rejected_count_by_type.get("fact", 0) >= 1
    assert telemetry.merge_candidate_similarity
    assert telemetry.merge_false_reject_rate >= 0.0
