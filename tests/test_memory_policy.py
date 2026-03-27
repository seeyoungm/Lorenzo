from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning.planner import ReasoningPlanner


def _build_orchestrator(path: Path) -> LorenzoOrchestrator:
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=JsonlMemoryStore(path),
        memory_retriever=MemoryRetriever(),
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(EchoBackend()),
    )
    return LorenzoOrchestrator(
        modules=modules,
        top_k=3,
        min_importance_to_store=6.0,
        dedup_similarity_threshold=0.90,
        semantic_merge_similarity_threshold=0.70,
    )


def test_question_is_stored_as_working_memory_only(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path / "memory.jsonl")

    orchestrator.run_turn("안녕?")

    memories = orchestrator.modules.memory_store.list_all()
    assert len(memories) == 1
    assert memories[0].memory_type.value == "working"


def test_duplicate_goal_does_not_unboundedly_grow_memory(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path / "memory.jsonl")

    text = "나는 장기 기억이 있는 AI를 만들고 싶어"
    orchestrator.run_turn(text)
    first_count = orchestrator.modules.memory_store.count()

    orchestrator.run_turn(text)
    second_count = orchestrator.modules.memory_store.count()

    assert first_count == 1
    assert second_count == 1

    memories = orchestrator.modules.memory_store.list_all()
    semantic_memories = [m for m in memories if m.memory_type.value == "semantic"]
    assert len(semantic_memories) == 1
    assert semantic_memories[0].metadata.get("seen_count", 0) >= 2
