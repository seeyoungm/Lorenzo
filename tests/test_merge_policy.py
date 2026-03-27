from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning.planner import ReasoningPlanner


def _build(path: Path, merge_confidence_threshold: float) -> LorenzoOrchestrator:
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
        semantic_merge_similarity_threshold=0.50,
        merge_confidence_threshold=merge_confidence_threshold,
    )


def test_semantic_merge_uses_summary_synthesis_and_tracks_success(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory.jsonl", merge_confidence_threshold=0.55)

    orch.run_turn("나는 장기 기억 중심의 모듈형 AI를 만들고 싶어")
    orch.run_turn("나는 장기 기억 중심의 모듈형 AI를 만들고 싶어, 핵심은 장기 메모리 모듈이야")

    items = orch.modules.memory_store.list_all()
    semantic = [item for item in items if item.memory_type.value == "semantic"]

    assert len(semantic) == 1
    assert "User goal:" in semantic[0].content
    assert "|" in semantic[0].content
    assert semantic[0].metadata.get("merge_policy") == "semantic_summary_synthesis"
    assert semantic[0].metadata.get("merged_from_ids")

    telemetry = orch.snapshot_telemetry()
    assert telemetry.merge_success_count >= 1
    assert telemetry.merge_rejected_count == 0


def test_low_confidence_merge_is_rejected_and_keeps_both_memories(tmp_path: Path) -> None:
    orch = _build(tmp_path / "memory.jsonl", merge_confidence_threshold=0.95)

    orch.run_turn("나는 장기 기억 중심 AI를 만들고 싶어")
    orch.run_turn("나는 기억 도우미를 만들고 싶어")

    items = orch.modules.memory_store.list_all()
    semantic = [item for item in items if item.memory_type.value == "semantic"]

    assert len(semantic) == 2
    telemetry = orch.snapshot_telemetry()
    assert telemetry.merge_rejected_count >= 1
