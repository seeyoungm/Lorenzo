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


def test_latest_recall_phrase_is_detected_as_memory_recall() -> None:
    processor = InputProcessor()

    recall = processor.process("What was my latest budget?")

    assert InputType.MEMORY_RECALL in recall.input_types
    assert InputType.QUESTION in recall.input_types


def test_goal_requires_target_state_or_long_term_signal() -> None:
    processor = InputProcessor()

    goal = processor.process("장기적으로 메모리 중심 AI 아키텍처를 완성하는 게 목표야")
    non_goal = processor.process("오늘 커피 마시고 싶어")

    assert InputType.GOAL_STATEMENT in goal.input_types
    assert InputType.GOAL_STATEMENT not in non_goal.input_types


def test_preference_is_style_behavior_not_commitment() -> None:
    processor = InputProcessor()

    preference = processor.process("답변 스타일은 간결한 bullet 형식을 선호해")
    commitment_like = processor.process("답변을 간결하게 줄게")

    assert InputType.PREFERENCE in preference.input_types
    assert InputType.COMMITMENT not in preference.input_types
    assert InputType.COMMITMENT not in commitment_like.input_types
    assert InputType.PREFERENCE not in commitment_like.input_types


def test_partial_ambiguous_fact_is_weak_fact_not_strong_fact() -> None:
    processor = InputProcessor()

    result = processor.process("예산은 아마 120 정도였던 것 같아")

    assert result.fact_confidence == "weak"
    assert InputType.FACT not in result.input_types


def test_recall_noise_phrase_is_not_weak_fact() -> None:
    processor = InputProcessor()

    result = processor.process("budget recall now")

    assert result.fact_confidence == "none"
    assert InputType.FACT not in result.input_types


def test_commitment_detects_explicit_promise_and_scheduled_future_action() -> None:
    processor = InputProcessor()

    explicit = processor.process("I will send the report tomorrow.")
    scheduled = processor.process("나는 내일까지 문서를 제출하겠다")
    promise_tone = processor.process("나는 약속한다, 보고서를 제출하겠다")

    assert InputType.COMMITMENT in explicit.input_types
    assert InputType.COMMITMENT in scheduled.input_types
    assert InputType.COMMITMENT in promise_tone.input_types


def test_commitment_takes_priority_over_goal_and_preference_when_overlapping() -> None:
    processor = InputProcessor()

    result = processor.process("나는 내일까지 문서를 제출할게, 이후에는 품질을 높이고 싶어")

    assert result.input_type is InputType.COMMITMENT


def test_vague_intention_is_not_commitment() -> None:
    processor = InputProcessor()

    result = processor.process("나는 내일 문서를 생각해볼게")

    assert InputType.COMMITMENT not in result.input_types


def test_speculative_statement_is_not_commitment() -> None:
    processor = InputProcessor()

    result = processor.process("아마 내일 문서를 제출할 수도 있어")

    assert InputType.COMMITMENT not in result.input_types


def test_commitment_requires_clear_subject() -> None:
    processor = InputProcessor()

    no_subject = processor.process("내일까지 문서를 제출하겠다")
    with_subject = processor.process("나는 내일까지 문서를 제출하겠다")

    assert InputType.COMMITMENT not in no_subject.input_types
    assert InputType.COMMITMENT in with_subject.input_types
