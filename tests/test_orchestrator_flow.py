from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.models import MemoryType
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning.planner import ReasoningPlanner


class FlagRetriever:
    def __init__(self) -> None:
        self.called = False

    def retrieve(self, query, memories, top_k=5):  # noqa: ANN001
        self.called = True
        return []

    def text_similarity(self, left, right):  # noqa: ANN001
        return 1.0 if left == right else 0.0


class AssertBackend:
    def __init__(self, retriever: FlagRetriever) -> None:
        self.retriever = retriever

    def generate(self, user_input, processed, retrieved, plan):  # noqa: ANN001
        assert self.retriever.called, "retrieval must happen before generation"
        return "ok"


def test_orchestrator_runs_retrieval_before_generation_and_updates_memory(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    retriever = FlagRetriever()
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=store,
        memory_retriever=retriever,
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(AssertBackend(retriever)),
    )
    orchestrator = LorenzoOrchestrator(modules=modules, top_k=3)

    result = orchestrator.run_turn("나는 장기 기억이 있는 AI를 만들고 싶어")

    assert result.response == "ok"
    memory_types = [item.memory_type for item in store.list_all()]
    assert memory_types.count(MemoryType.SEMANTIC) == 1
    assert MemoryType.EPISODIC not in memory_types


def test_pure_question_is_working_only(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    retriever = FlagRetriever()
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=store,
        memory_retriever=retriever,
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(AssertBackend(retriever)),
    )
    orchestrator = LorenzoOrchestrator(modules=modules, top_k=3)

    orchestrator.run_turn("오늘 날씨 어때?")

    memory_types = [item.memory_type for item in store.list_all()]
    assert memory_types == [MemoryType.WORKING]
