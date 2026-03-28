from __future__ import annotations

from pathlib import Path

import pytest

from lorenzo.input_processor import InputProcessor
from lorenzo.language.adapter import LanguageAdapter
from lorenzo.language.backends import EchoBackend
from lorenzo.memory.retriever import MemoryRetriever
from lorenzo.memory.store import JsonlMemoryStore
from lorenzo.models import MemoryType
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
    ("text", "expected_type", "expected_tag"),
    [
        ("장기적으로 메모리 중심 AI를 완성하는 것이 목표다", MemoryType.SEMANTIC, "goal"),
        ("답변 스타일은 간결한 bullet을 선호해", MemoryType.SEMANTIC, "preference"),
        ("나는 내일까지 보고서를 제출하겠다", MemoryType.SEMANTIC, "commitment"),
        ("중요: 예산은 100만원입니다", MemoryType.SEMANTIC, "fact"),
        ("내일 오후 3시 회의가 있어", MemoryType.EPISODIC, "event"),
    ],
)
def test_storage_type_alignment_diverse(
    tmp_path: Path,
    text: str,
    expected_type: MemoryType,
    expected_tag: str,
) -> None:
    orch = _build(tmp_path / "aligned.jsonl")
    orch.run_turn(text)

    items = orch.modules.memory_store.list_all()
    assert items
    assert any(item.memory_type is expected_type and expected_tag in [t.lower() for t in item.tags] for item in items)


@pytest.mark.parametrize(
    "text",
    [
        "나는 내일 문서를 생각해볼게",
        "아마 내일 제출할 수도 있어",
        "이렇게 답변할게",
    ],
)
def test_negative_commitment_cases_are_not_stored_as_commitment(tmp_path: Path, text: str) -> None:
    orch = _build(tmp_path / "negative.jsonl")
    orch.run_turn(text)

    items = orch.modules.memory_store.list_all()
    assert all("commitment" not in [t.lower() for t in item.tags] for item in items)


def test_weak_preference_and_weak_fact_are_stored_as_weak_hints(tmp_path: Path) -> None:
    orch = _build(tmp_path / "weak-hints.jsonl")

    orch.run_turn("가능하면 오늘은 답변 톤을 조금 더 짧게 해줘")
    orch.run_turn("예산은 아마 120 정도였던 것 같아")

    items = orch.modules.memory_store.list_all()
    weak_preference = [
        item
        for item in items
        if item.content.startswith("Weak preference hint:")
    ]
    weak_fact = [item for item in items if item.content.startswith("Weak fact hint:")]

    assert len(weak_preference) == 1
    assert weak_preference[0].metadata.get("memory_tier") == "weak"
    assert "weak_preference_hint" in [tag.lower() for tag in weak_preference[0].tags]

    assert len(weak_fact) == 1
    assert weak_fact[0].metadata.get("memory_tier") == "weak"
    assert "weak_fact_hint" in [tag.lower() for tag in weak_fact[0].tags]
    assert weak_fact[0].metadata.get("fact_key") == "budget"
    assert weak_fact[0].metadata.get("fact_value") == "120"
