from __future__ import annotations

import pytest

from lorenzo.input_processor import InputProcessor
from lorenzo.models import InputType


@pytest.mark.parametrize(
    ("text", "must_have", "must_not_have", "primary"),
    [
        (
            "장기적으로 모듈형 기억 아키텍처를 완성하는 게 목표다",
            {InputType.GOAL_STATEMENT},
            {InputType.COMMITMENT},
            InputType.GOAL_STATEMENT,
        ),
        (
            "I would like to build a long-term memory AI platform.",
            {InputType.GOAL_STATEMENT},
            {InputType.COMMITMENT},
            InputType.GOAL_STATEMENT,
        ),
        (
            "오늘 커피 마시고 싶어",
            set(),
            {InputType.GOAL_STATEMENT, InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "답변 스타일은 짧고 구조화된 bullet을 선호해",
            {InputType.PREFERENCE},
            {InputType.COMMITMENT},
            InputType.PREFERENCE,
        ),
        (
            "I prefer concise and structured answers.",
            {InputType.PREFERENCE},
            {InputType.COMMITMENT},
            InputType.PREFERENCE,
        ),
        (
            "나는 내일까지 보고서를 제출하겠다",
            {InputType.COMMITMENT},
            set(),
            InputType.COMMITMENT,
        ),
        (
            "I will send the final report by tomorrow.",
            {InputType.COMMITMENT},
            set(),
            InputType.COMMITMENT,
        ),
        (
            "나는 약속한다, 이번 주 안에 테스트를 완료하겠다",
            {InputType.COMMITMENT},
            set(),
            InputType.COMMITMENT,
        ),
        (
            "나는 내일 문서를 생각해볼게",
            set(),
            {InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "아마 내일 제출할 수도 있어",
            set(),
            {InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "이렇게 답변할게",
            set(),
            {InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "답변을 간결하게 줄게",
            set(),
            {InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "내일까지 문서를 제출하겠다",
            set(),
            {InputType.COMMITMENT},
            InputType.STATEMENT,
        ),
        (
            "나는 내일까지 문서를 제출할게, 이후 목표를 정리하고 싶어",
            {InputType.COMMITMENT},
            set(),
            InputType.COMMITMENT,
        ),
        (
            "내일 9시?",
            {InputType.MEMORY_CANDIDATE, InputType.EVENT, InputType.QUESTION},
            set(),
            InputType.EVENT,
        ),
        (
            "예산이 얼마였지?",
            {InputType.MEMORY_RECALL, InputType.QUESTION},
            set(),
            InputType.MEMORY_RECALL,
        ),
    ],
)
def test_intent_boundaries_matrix(
    text: str,
    must_have: set[InputType],
    must_not_have: set[InputType],
    primary: InputType,
) -> None:
    processor = InputProcessor()
    result = processor.process(text)

    for item in must_have:
        assert item in result.input_types
    for item in must_not_have:
        assert item not in result.input_types
    assert result.input_type is primary


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("나는 이번 주 안에 테스트 코드를 제출하겠다", True),
        ("I will finish the task by Friday.", True),
        ("나는 곧 해볼게", False),
        ("maybe I will submit it tomorrow", False),
        ("답변은 이렇게 할게", False),
        ("내일까지 완료", False),
    ],
)
def test_commitment_precision_recall_edges(text: str, expected: bool) -> None:
    processor = InputProcessor()
    result = processor.process(text)
    detected = InputType.COMMITMENT in result.input_types
    assert detected is expected
