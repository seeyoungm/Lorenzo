from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning.planner import ReasoningPlanner


def _build(path: Path) -> LorenzoOrchestrator:
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=JsonlMemoryStore(path),
        memory_retriever=MemoryRetriever(),
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(EchoBackend()),
    )
    return LorenzoOrchestrator(modules=modules, top_k=5)


def test_conflicting_facts_use_latest_wins_policy(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory.jsonl")

    orch.run_turn("중요: 프로젝트 예산은 100만원입니다")
    orch.run_turn("중요: 프로젝트 예산은 30만원입니다")

    items = orch.modules.memory_store.list_all()
    fact_items = [item for item in items if item.content.startswith("User fact:")]
    assert len(fact_items) >= 1

    latest = fact_items[0]
    assert "30만원" in latest.content
    assert latest.metadata.get("conflict_policy") == "latest_wins"
    assert latest.metadata.get("conflict_history")

    telemetry = orch.snapshot_telemetry()
    assert telemetry.conflict_resolved >= 1


def test_conflicting_events_use_latest_wins_policy(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory_events.jsonl")

    orch.run_turn("내일 회의는 오후 3시야")
    orch.run_turn("회의가 내일 오후 5시로 변경됐어")

    items = orch.modules.memory_store.list_all()
    event_items = [item for item in items if item.content.startswith("User event:")]
    assert len(event_items) >= 1

    latest = event_items[0]
    assert "5시" in latest.content
    assert latest.metadata.get("conflict_policy") == "latest_wins"
