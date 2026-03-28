from __future__ import annotations

from dataclasses import dataclass
import re

from lorenzo.models import InputType, ProcessedInput


@dataclass(slots=True)
class Claim:
    claim_type: str
    text: str
    key: str = ""
    value: str = ""


class ClaimExtractor:
    def __init__(self) -> None:
        self._fact_key_aliases = {
            "budget": ("budget", "예산", "비용", "cost"),
            "deadline": ("deadline", "마감", "마감일", "due"),
            "price": ("price", "가격", "요금"),
            "version": ("version", "버전", "release"),
            "rate": ("rate", "비율", "percent", "%"),
        }
        self._preference_tokens = ["preference", "prefer", "선호", "원해", "좋아", "싫어"]
        self._preference_style_tokens = {
            "concise": ("간결", "짧", "concise", "short"),
            "detailed": ("자세", "길게", "detailed", "long"),
        }
        self._preference_format_tokens = {
            "bullet": ("bullet", "불릿", "목록"),
            "no_bullet": ("문단", "paragraph", "no bullet", "불릿 말고"),
        }
        self._preference_tone_tokens = {
            "formal": ("정중", "공손", "formal", "polite"),
            "casual": ("편하게", "캐주얼", "casual", "친근"),
        }
        self._goal_tokens = ("goal", "목표", "계획", "장기", "장기적", "완성", "달성", "구축")
        self._goal_future_tokens = ("할", "하겠다", "완성하겠다", "달성하겠다", "will", "plan", "by ")
        self._goal_achievement_tokens = ("완성", "달성", "구축", "출시", "릴리스", "ship", "deliver")
        self._goal_weak_tokens = (
            "좋겠다",
            "wish",
            "바람",
            "생각",
            "의견",
            "해볼게",
            "생각해볼게",
            "언젠가",
            "maybe",
        )

    def extract_claims(self, draft_answer: str, user_input: str, processed: ProcessedInput) -> list[Claim]:
        answer_text = self._extract_answer_body(draft_answer)
        query_text = user_input.strip()
        question_like = (
            InputType.QUESTION in processed.input_types
            or InputType.MEMORY_RECALL in processed.input_types
        )

        claims: list[Claim] = []
        claims.extend(self._extract_fact_claims(answer_text))
        claims.extend(self._extract_preference_claims(answer_text))
        claims.extend(self._extract_goal_claims(answer_text))

        # Intent-aware fallback is allowed only for near-empty drafts to avoid query over-extraction.
        if (not claims) and (len(answer_text.strip()) < 12) and (not question_like):
            if InputType.FACT in processed.input_types:
                claims.extend(self._extract_fact_claims(query_text))
            if InputType.PREFERENCE in processed.input_types:
                claims.extend(self._extract_preference_claims(query_text))
            if InputType.GOAL_STATEMENT in processed.input_types:
                claims.extend(self._extract_goal_claims(query_text))

        if (not claims) and (not question_like):
            generic = answer_text.strip() or processed.raw_text.strip()
            if generic:
                claims.append(Claim(claim_type="generic", text=generic[:180]))

        deduped = self._dedup_claims(claims)
        return sorted(deduped, key=self._sort_key)

    def _extract_answer_body(self, raw: str) -> str:
        lines: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                continue
            if stripped.startswith("참고 기억"):
                continue
            # Memory evidence list is not a direct answer claim.
            if stripped.startswith("- "):
                continue
            lines.append(stripped)
        return "\n".join(lines).strip()

    def _extract_fact_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        if not lowered:
            return []

        claims: list[Claim] = []
        for match in re.finditer(r"key=([a-z_]+)\s*;\s*value=([^;\n]+)", lowered):
            key = match.group(1).strip()
            value = self._normalize_value(match.group(2))
            if key and value:
                claims.append(
                    Claim(
                        claim_type="fact",
                        text=match.group(0).strip(),
                        key=key,
                        value=value,
                    )
                )

        segments = re.split(r"[.\n;!?]", text)
        for segment in segments:
            sentence = segment.strip()
            if not sentence:
                continue
            sentence_lower = sentence.lower()
            for canonical_key, aliases in self._fact_key_aliases.items():
                alias = next((token for token in aliases if token in sentence_lower), "")
                if not alias:
                    continue

                value = self._value_near_alias(sentence_lower, alias)
                if value:
                    claims.append(
                        Claim(
                            claim_type="fact",
                            text=sentence[:120],
                            key=canonical_key,
                            value=value,
                        )
                    )
        return claims

    def _value_near_alias(self, sentence_lower: str, alias: str) -> str:
        idx = sentence_lower.find(alias)
        if idx < 0:
            return ""

        window_start = max(0, idx - 24)
        window_end = min(len(sentence_lower), idx + len(alias) + 24)
        near_text = sentence_lower[window_start:window_end]
        num = re.search(r"\d+(?:[.,]\d+)?(?::\d{2})?", near_text)
        if num:
            return num.group(0).replace(",", "")

        # Non-numeric fact values are allowed only when explicitly tied to a key.
        explicit = re.search(
            rf"{re.escape(alias)}\s*(?:is|was|=|은|는|이|가)\s*([a-z0-9가-힣 _-]{{2,24}})",
            sentence_lower,
        )
        if explicit:
            candidate = explicit.group(1).strip()
            if candidate not in {"unknown", "모름"}:
                return candidate
        return ""

    def _extract_preference_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        if not lowered:
            return []

        has_preference_signal = any(token in lowered for token in self._preference_tokens)
        style_hits = self._match_preference_value(lowered, self._preference_style_tokens)
        format_hits = self._match_preference_value(lowered, self._preference_format_tokens)
        tone_hits = self._match_preference_value(lowered, self._preference_tone_tokens)
        if not (has_preference_signal or style_hits or format_hits or tone_hits):
            return []

        claims: list[Claim] = []
        for value in style_hits:
            claims.append(
                Claim(
                    claim_type="preference",
                    text=f"style={value}",
                    key="style",
                    value=value,
                )
            )
        for value in format_hits:
            claims.append(
                Claim(
                    claim_type="preference",
                    text=f"format={value}",
                    key="format",
                    value=value,
                )
            )
        for value in tone_hits:
            claims.append(
                Claim(
                    claim_type="preference",
                    text=f"tone={value}",
                    key="tone",
                    value=value,
                )
            )

        if not claims and has_preference_signal:
            claims.append(
                Claim(
                    claim_type="preference",
                    text="preference mentioned",
                    key="style",
                    value="generic",
                )
            )
        return claims

    def _extract_goal_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        if not lowered:
            return []
        if not any(token in lowered for token in self._goal_tokens):
            return []

        weak_goal = any(token in lowered for token in self._goal_weak_tokens)
        future_orientation = any(token in lowered for token in self._goal_future_tokens)
        achievement_intent = any(token in lowered for token in self._goal_achievement_tokens)
        if weak_goal or (not future_orientation) or (not achievement_intent):
            return []

        snippet = text.strip().split("\n")[0][:120]
        return [Claim(claim_type="goal", text=snippet, key="goal", value="strong")]

    def _match_preference_value(
        self,
        text: str,
        mapping: dict[str, tuple[str, ...]],
    ) -> list[str]:
        values: list[str] = []
        for key, tokens in mapping.items():
            if any(token in text for token in tokens):
                values.append(key)
        return values

    def _normalize_value(self, raw: str) -> str:
        value = raw.strip().lower()
        numeric = re.search(r"\d+(?:\.\d+)?", value)
        if numeric:
            return numeric.group(0)
        return value

    def _dedup_claims(self, claims: list[Claim]) -> list[Claim]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[Claim] = []
        for claim in claims:
            key = (
                claim.claim_type.strip().lower(),
                claim.key.strip().lower(),
                claim.value.strip().lower(),
                claim.text.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)
        return deduped

    def _sort_key(self, claim: Claim) -> tuple[int, int, str, str, str]:
        type_priority = {"fact": 0, "preference": 1, "goal": 2, "generic": 3}
        specificity = 0
        if claim.key:
            specificity -= 1
        if claim.value:
            specificity -= 1
        return (
            type_priority.get(claim.claim_type, 99),
            specificity,
            claim.key,
            claim.value,
            claim.text,
        )
