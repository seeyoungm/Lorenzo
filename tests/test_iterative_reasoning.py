from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory
from lorenzo.orchestrator import LorenzoOrchestrator, PipelineModules
from lorenzo.reasoning import IterativeReasoningEngine
from lorenzo.reasoning.planner import ReasoningPlanner


class TwoPassRetriever:
    def __init__(self) -> None:
        self.calls = 0
        now = datetime(2026, 3, 28, tzinfo=timezone.utc)
        self.low_support = RetrievedMemory(
            memory=MemoryItem(
                content="Weak fact hint: key=budget; value=90; source=budget maybe 90",
                memory_type=MemoryType.SEMANTIC,
                tags=["fact", "semantic", "weak", "weak_hint", "weak_fact_hint"],
                metadata={"memory_tier": "weak", "fact_key": "budget", "fact_value": "90"},
                created_at=now,
            ),
            total_score=0.18,
            similarity_score=0.10,
            recency_score=0.90,
            importance_score=0.40,
            retrieval_reason="first-pass",
        )
        self.high_support = RetrievedMemory(
            memory=MemoryItem(
                content="User fact: 프로젝트 예산은 120만원이다",
                memory_type=MemoryType.SEMANTIC,
                tags=["fact", "semantic", "strong"],
                metadata={"memory_tier": "strong"},
                created_at=now,
            ),
            total_score=0.92,
            similarity_score=0.88,
            recency_score=0.95,
            importance_score=0.85,
            retrieval_reason="second-pass",
        )

    def retrieve(self, query, memories, top_k=5, now=None, mode="auto"):  # noqa: ANN001
        self.calls += 1
        if "Draft answer for refinement" in query:
            return [self.high_support]
        return [self.low_support]

    def text_similarity(self, left, right):  # noqa: ANN001
        return 1.0 if left == right else 0.0


class TraceBackend:
    def generate(self, user_input, processed, retrieved, plan):  # noqa: ANN001
        if retrieved:
            return f"answer<{retrieved[0].memory.content}>"
        return "answer<none>"


def test_iterative_reasoning_refines_answer_when_support_improves(tmp_path: Path) -> None:
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    retriever = TwoPassRetriever()
    modules = PipelineModules(
        input_processor=InputProcessor(),
        memory_store=store,
        memory_retriever=retriever,
        reasoning_planner=ReasoningPlanner(),
        language_adapter=LanguageAdapter(TraceBackend()),
    )
    orchestrator = LorenzoOrchestrator(modules=modules, top_k=3, iterative_reasoning_max_iterations=2)

    result = orchestrator.run_turn("현재 예산 뭐였지?")
    telemetry = orchestrator.snapshot_telemetry()

    assert retriever.calls >= 2
    assert "User fact: 프로젝트 예산은 120만원이다" in result.response
    assert telemetry.refinement_attempts == 1
    assert telemetry.refinement_improvement_count == 1
    assert telemetry.refinement_answer_changed_count == 1


def test_iterative_reasoning_engine_iteration_is_bounded() -> None:
    engine = IterativeReasoningEngine(max_iterations=9)

    assert engine.max_iterations == 3
