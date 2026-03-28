from __future__ import annotations

from dataclasses import dataclass
import re

from lorenzo.models import ProcessedInput


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
            "version": ("version", "버전", "릴리스"),
        }
        self._preference_tokens = ["preference", "prefer", "선호", "스타일", "style", "tone", "형식"]
        self._preference_concise_tokens = ["간결", "짧", "concise", "short", "bullet"]
        self._preference_detailed_tokens = ["자세", "길게", "detailed", "long"]
        self._goal_tokens = [
            "goal",
            "목표",
            "장기",
            "long-term",
            "비전",
            "달성",
            "완성",
            "만들",
            "구축",
        ]
        self._goal_exclusion_tokens = ["wish", "바람", "좋겠다", "의견", "opinion", "생각"]

    def extract_claims(self, draft_answer: str, user_input: str, processed: ProcessedInput) -> list[Claim]:
        claims: list[Claim] = []
        answer_text = draft_answer.strip()
        query_text = user_input.strip()
        combined_text = f"{answer_text}\n{query_text}".strip()

        claims.extend(self._extract_fact_claims(combined_text))
        claims.extend(self._extract_preference_claims(combined_text))
        claims.extend(self._extract_goal_claims(combined_text))

        # If explicit claim was not found, keep at least one generic claim for support tracking.
        if not claims:
            if answer_text:
                claims.append(Claim(claim_type="generic", text=answer_text[:180]))
            elif processed.raw_text.strip():
                claims.append(Claim(claim_type="generic", text=processed.raw_text.strip()[:180]))

        deduped = self._dedup_claims(claims)
        return sorted(
            deduped,
            key=lambda claim: (
                self._claim_priority(claim.claim_type),
                claim.key,
                claim.value,
                claim.text,
            ),
        )

    def _extract_fact_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        claims: list[Claim] = []

        structured = re.finditer(r"key=([a-z_]+)\s*;\s*value=([^;\n]+)", lowered)
        for match in structured:
            key = match.group(1).strip()
            value = self._normalize_value(match.group(2))
            claims.append(Claim(claim_type="fact", text=match.group(0).strip(), key=key, value=value))

        numeric_values = re.findall(r"\d+(?:\.\d+)?", lowered)
        for canonical_key, aliases in self._fact_key_aliases.items():
            if any(alias in lowered for alias in aliases):
                if numeric_values:
                    for number in dict.fromkeys(numeric_values):
                        claims.append(
                            Claim(
                                claim_type="fact",
                                text=f"{canonical_key}={number}",
                                key=canonical_key,
                                value=number,
                            )
                        )
                else:
                    claims.append(Claim(claim_type="fact", text=f"{canonical_key}=unknown", key=canonical_key))
        return claims

    def _extract_preference_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        if not any(token in lowered for token in self._preference_tokens):
            return []

        claims: list[Claim] = []
        if any(token in lowered for token in self._preference_concise_tokens):
            claims.append(Claim(claim_type="preference", text="prefer concise style", key="style", value="concise"))
        if any(token in lowered for token in self._preference_detailed_tokens):
            claims.append(Claim(claim_type="preference", text="prefer detailed style", key="style", value="detailed"))
        if not claims:
            claims.append(Claim(claim_type="preference", text="preference mentioned", key="style", value="generic"))
        return claims

    def _extract_goal_claims(self, text: str) -> list[Claim]:
        lowered = text.lower()
        if not any(token in lowered for token in self._goal_tokens):
            return []
        if any(token in lowered for token in self._goal_exclusion_tokens):
            return []

        snippet = text.strip().split("\n")[0][:120]
        return [Claim(claim_type="goal", text=snippet, key="goal", value="present")]

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
            key = (claim.claim_type, claim.key, claim.value, claim.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)
        return deduped

    def _claim_priority(self, claim_type: str) -> int:
        priorities = {"fact": 0, "preference": 1, "goal": 2, "generic": 3}
        return priorities.get(claim_type, 99)
