from lorenzo.input_processor import InputProcessor
from lorenzo.models import InputType


def test_memory_command_is_not_overridden_by_question_mark() -> None:
    processor = InputProcessor()

    result = processor.process("기억해?")

    assert result.input_type is InputType.MEMORY_CANDIDATE
    assert InputType.MEMORY_CANDIDATE in result.input_types
    assert InputType.QUESTION in result.input_types
    assert InputType.MEMORY_INSTRUCTION in result.input_types


def test_keyword_based_memory_trigger_english() -> None:
    processor = InputProcessor()

    result = processor.process("remember this?")

    assert result.input_type is InputType.MEMORY_CANDIDATE
    assert InputType.MEMORY_CANDIDATE in result.input_types
    assert InputType.QUESTION in result.input_types


def test_short_time_question_is_memory_candidate_and_event() -> None:
    processor = InputProcessor()

    result = processor.process("내일 9시?")

    assert result.input_type is InputType.EVENT
    assert InputType.EVENT in result.input_types
    assert InputType.MEMORY_CANDIDATE in result.input_types
    assert InputType.QUESTION in result.input_types


def test_regular_question_remains_question() -> None:
    processor = InputProcessor()

    result = processor.process("오늘 날씨 어때?")

    assert result.input_type is InputType.QUESTION
    assert InputType.QUESTION in result.input_types
    assert InputType.MEMORY_CANDIDATE not in result.input_types


def test_fact_and_commitment_detection() -> None:
    processor = InputProcessor()

    fact = processor.process("중요: 예산은 100만원입니다")
    commitment = processor.process("나는 내일까지 문서를 제출할게")

    assert InputType.FACT in fact.input_types
    assert InputType.COMMITMENT in commitment.input_types


def test_memory_recall_detection() -> None:
    processor = InputProcessor()

    recall = processor.process("예산이 얼마였지?")

    assert InputType.MEMORY_RECALL in recall.input_types
    assert InputType.QUESTION in recall.input_types
