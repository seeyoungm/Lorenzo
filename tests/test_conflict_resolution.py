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


def test_conflicting_facts_keep_both_and_mark_conflict(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory.jsonl")

    orch.run_turn("중요: 프로젝트 예산은 100만원입니다")
    orch.run_turn("중요: 프로젝트 예산은 30만원입니다")

    items = orch.modules.memory_store.list_all()
    fact_items = [item for item in items if item.content.startswith("User fact:")]
    assert len(fact_items) >= 2
    assert any("100만원" in item.content for item in fact_items)
    assert any("30만원" in item.content for item in fact_items)
    assert any(item.metadata.get("conflict_strategy") == "keep_both_mark_conflict" for item in fact_items)
    assert any(item.metadata.get("conflict_reason") == "fact_value_conflict" for item in fact_items)

    telemetry = orch.snapshot_telemetry()
    assert telemetry.conflict_resolved >= 1


def test_events_preserve_separately_by_timestamp(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory_events.jsonl")

    orch.run_turn("내일 회의는 오후 3시야")
    orch.run_turn("회의가 내일 오후 5시로 변경됐어")

    items = orch.modules.memory_store.list_all()
    event_items = [item for item in items if item.content.startswith("User event:")]
    assert len(event_items) == 2
    assert all(item.metadata.get("event_timestamp") for item in event_items)
    assert all(item.metadata.get("conflict_strategy") == "preserve_by_timestamp" for item in event_items)
