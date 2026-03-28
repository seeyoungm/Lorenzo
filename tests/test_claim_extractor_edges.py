from __future__ import annotations

from lorenzo.models import InputType, ProcessedInput
from lorenzo.reasoning.claim_extractor import ClaimExtractor


def _processed(text: str, input_types: list[InputType]) -> ProcessedInput:
    return ProcessedInput(
        raw_text=text,
        input_type=input_types[0] if input_types else InputType.STATEMENT,
        input_types=input_types,
    )


def test_numeric_text_without_fact_key_is_not_over_extracted_as_fact() -> None:
    extractor = ClaimExtractor()
    text = "총 3가지 선택지를 정리해볼게."

    claims = extractor.extract_claims(
        draft_answer=text,
        user_input=text,
        processed=_processed(text, [InputType.STATEMENT]),
    )

    assert all(claim.claim_type != "fact" for claim in claims)


def test_goal_extraction_distinguishes_wish_vs_strong_goal() -> None:
    extractor = ClaimExtractor()

    wish_text = "언젠가 이런 시스템이면 좋겠어."
    wish_claims = extractor.extract_claims(
        draft_answer=wish_text,
        user_input=wish_text,
        processed=_processed(wish_text, [InputType.GOAL_STATEMENT]),
    )
    assert all(claim.claim_type != "goal" for claim in wish_claims)

    strong_text = "올해 안에 모듈형 기억 시스템을 완성하겠다."
    strong_claims = extractor.extract_claims(
        draft_answer=strong_text,
        user_input=strong_text,
        processed=_processed(strong_text, [InputType.GOAL_STATEMENT]),
    )
    assert any(claim.claim_type == "goal" for claim in strong_claims)


def test_preference_extraction_supports_format_and_tone() -> None:
    extractor = ClaimExtractor()
    text = "답변은 bullet 말고 문단으로, 톤은 정중하게 해줘."

    claims = extractor.extract_claims(
        draft_answer=text,
        user_input=text,
        processed=_processed(text, [InputType.PREFERENCE]),
    )
    kv = {(claim.key, claim.value) for claim in claims if claim.claim_type == "preference"}

    assert ("format", "no_bullet") in kv
    assert ("tone", "formal") in kv
