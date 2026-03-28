from __future__ import annotations

from pathlib import Path

import pytest

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.models import InputType
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
    return LorenzoOrchestrator(modules=modules, top_k=5, min_importance_to_store=6.5)


@pytest.mark.parametrize(
    ("text", "expected_confidence", "should_have_goal_label"),
    [
        ("장기적으로 메모리 중심 AI 제품을 완성하는 게 목표야", "strong", True),
        ("I would like to build a long-term memory AI platform.", "strong", True),
        ("나는 장기 기억이 있는 AI를 만들고 싶어", "strong", True),
        ("목표를 한번 생각해볼게", "weak", False),
        ("언젠가 AI 플랫폼을 만들어보고 싶어", "weak", False),
        ("오늘 당장 작은 데모를 만들고 싶어", "weak", False),
        ("장기적으로 이런 AI가 되면 좋겠어", "none", False),
        ("내 생각에는 이 방향이 더 좋아 보여", "none", False),
        ("이렇게 답변하면 좋겠다", "none", False),
    ],
)
def test_goal_confidence_refinement_boundaries(
    text: str,
    expected_confidence: str,
    should_have_goal_label: bool,
) -> None:
    processor = InputProcessor()
    result = processor.process(text)

    assert result.goal_confidence == expected_confidence
    if should_have_goal_label:
        assert InputType.GOAL_STATEMENT in result.input_types
    else:
        assert InputType.GOAL_STATEMENT not in result.input_types


def test_only_strong_goal_is_stored_as_semantic_goal_memory(tmp_path: Path) -> None:
    orch = _build(tmp_path / "goal-refine.jsonl")

    orch.run_turn("장기적으로 메모리 중심 AI 제품을 완성하는 게 목표야")
    orch.run_turn("목표를 한번 생각해볼게")
    orch.run_turn("오늘 당장 작은 데모를 만들고 싶어")

    goals = [
        item
        for item in orch.modules.memory_store.list_all()
        if item.content.startswith("User goal:")
    ]
    assert len(goals) == 1
    assert "완성" in goals[0].content


def test_goal_sentence_with_memory_noun_is_not_memory_instruction() -> None:
    processor = InputProcessor()
    result = processor.process("장기적으로 모듈형 기억 아키텍처를 완성하는 게 목표다")

    assert result.goal_confidence == "strong"
    assert InputType.GOAL_STATEMENT in result.input_types
    assert InputType.MEMORY_INSTRUCTION not in result.input_types
