from __future__ import annotations

from dataclasses import dataclass
import re

from lorenzo.models import MemoryType, RetrievedMemory
from lorenzo.reasoning.claim_extractor import Claim


@dataclass(slots=True)
class ClaimAssessment:
    claim: Claim
    status: str  # supported | contradicted | unsupported
    evidence_memory_ids: list[str]
    matched_memory_count: int = 0
    evidence_strength: float = 0.0


@dataclass(slots=True)
class VerificationSummary:
    assessments: list[ClaimAssessment]
    support_coverage: float
    unsupported_rate: float
    preference_alignment_score: float
    supported_count: int
    contradicted_count: int
    unsupported_count: int
    avg_evidence_strength: float


@dataclass(slots=True)
class RefinementAcceptPolicy:
    min_support_coverage_gain: float = 0.03
    min_preference_alignment_gain: float = 0.10
    min_evidence_strength_gain: float = 0.05
    max_support_coverage_drop: float = 0.02
    max_contradiction_increase: int = 0
    max_unsupported_increase: int = 0


@dataclass(slots=True)
class JudgeDecision:
    apply_refinement: bool
    improvement: bool
    regression: bool
    retrieval_improved_but_answer_worsened: bool
    support_coverage_increased: bool
    contradiction_reduced: bool
    unsupported_reduced: bool
    preference_alignment_improved: bool
    evidence_strength_improved: bool
    unsupported_claims_remaining: bool
    contradiction_persisted: bool
    support_not_improved: bool
    prefer_conservative_rewrite: bool
    reasons: list[str]


class ClaimAwareRefinementJudge:
    def __init__(self, policy: RefinementAcceptPolicy | None = None) -> None:
        self.policy = policy or RefinementAcceptPolicy()

    def verify_claims(self, claims: list[Claim], retrieved: list[RetrievedMemory]) -> VerificationSummary:
        assessments: list[ClaimAssessment] = []
        for claim in claims:
            status, evidence, strength = self._verify_claim(claim, retrieved)
            assessments.append(
                ClaimAssessment(
                    claim=claim,
                    status=status,
                    evidence_memory_ids=evidence,
                    matched_memory_count=len(evidence),
                    evidence_strength=strength,
                )
            )

        total = len(assessments)
        supported_count = sum(1 for item in assessments if item.status == "supported")
        contradicted_count = sum(1 for item in assessments if item.status == "contradicted")
        unsupported_count = sum(1 for item in assessments if item.status == "unsupported")
        support_coverage = supported_count / total if total > 0 else 0.0
        unsupported_rate = unsupported_count / total if total > 0 else 0.0
        preference_alignment_score = self._preference_alignment_score(assessments)
        avg_evidence_strength = (
            sum(item.evidence_strength for item in assessments) / total if total > 0 else 0.0
        )

        return VerificationSummary(
            assessments=assessments,
            support_coverage=support_coverage,
            unsupported_rate=unsupported_rate,
            preference_alignment_score=preference_alignment_score,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unsupported_count=unsupported_count,
            avg_evidence_strength=avg_evidence_strength,
        )

    def judge_refinement(
        self,
        *,
        draft_summary: VerificationSummary,
        refined_summary: VerificationSummary,
        draft_support_score: float,
        refined_support_score: float,
        support_gain_margin: float,
    ) -> JudgeDecision:
        support_gain = refined_summary.support_coverage - draft_summary.support_coverage
        preference_gain = (
            refined_summary.preference_alignment_score - draft_summary.preference_alignment_score
        )
        evidence_gain = refined_summary.avg_evidence_strength - draft_summary.avg_evidence_strength
        contradiction_change = refined_summary.contradicted_count - draft_summary.contradicted_count
        unsupported_change = refined_summary.unsupported_count - draft_summary.unsupported_count

        support_coverage_increased = support_gain > self.policy.min_support_coverage_gain
        contradiction_reduced = contradiction_change < 0
        unsupported_reduced = unsupported_change < 0
        preference_alignment_improved = preference_gain > self.policy.min_preference_alignment_gain
        evidence_strength_improved = evidence_gain > self.policy.min_evidence_strength_gain

        quality_worsened = (
            support_gain < -self.policy.max_support_coverage_drop
            or contradiction_change > self.policy.max_contradiction_increase
            or unsupported_change > self.policy.max_unsupported_increase
            or evidence_gain < -0.08
        )
        retrieval_improved = refined_support_score > (draft_support_score + support_gain_margin)
        retrieval_improved_but_answer_worsened = retrieval_improved and quality_worsened
        unsupported_claims_remaining = refined_summary.unsupported_count > 0
        contradiction_persisted = refined_summary.contradicted_count > 0
        support_not_improved = refined_summary.support_coverage <= draft_summary.support_coverage

        positive_signals = [
            support_coverage_increased,
            contradiction_reduced,
            unsupported_reduced,
            preference_alignment_improved,
            evidence_strength_improved,
        ]
        apply_refinement = any(positive_signals) and not quality_worsened
        if (not apply_refinement) and retrieval_improved and (not quality_worsened):
            apply_refinement = True

        prefer_conservative_rewrite = (
            retrieval_improved_but_answer_worsened
            or unsupported_claims_remaining
            or contradiction_persisted
            or support_not_improved
        )

        reasons: list[str] = []
        if support_coverage_increased:
            reasons.append("support_coverage_increased")
        if contradiction_reduced:
            reasons.append("contradiction_reduced")
        if unsupported_reduced:
            reasons.append("unsupported_claims_reduced")
        if preference_alignment_improved:
            reasons.append("preference_alignment_improved")
        if evidence_strength_improved:
            reasons.append("evidence_strength_improved")
        if unsupported_claims_remaining:
            reasons.append("unsupported_claims_remaining")
        if contradiction_persisted:
            reasons.append("contradiction_persisted")
        if quality_worsened:
            reasons.append("quality_worsened")
        if retrieval_improved_but_answer_worsened:
            reasons.append("retrieval_improved_but_answer_worsened")

        return JudgeDecision(
            apply_refinement=apply_refinement,
            improvement=apply_refinement and (not quality_worsened),
            regression=quality_worsened,
            retrieval_improved_but_answer_worsened=retrieval_improved_but_answer_worsened,
            support_coverage_increased=support_coverage_increased,
            contradiction_reduced=contradiction_reduced,
            unsupported_reduced=unsupported_reduced,
            preference_alignment_improved=preference_alignment_improved,
            evidence_strength_improved=evidence_strength_improved,
            unsupported_claims_remaining=unsupported_claims_remaining,
            contradiction_persisted=contradiction_persisted,
            support_not_improved=support_not_improved,
            prefer_conservative_rewrite=prefer_conservative_rewrite,
            reasons=reasons,
        )

    def has_conflicting_memories(self, retrieved: list[RetrievedMemory]) -> bool:
        grouped: dict[str, set[str]] = {}
        for item in retrieved:
            memory = item.memory
            if memory.memory_type is not MemoryType.SEMANTIC:
                continue
            if "fact" not in {tag.lower() for tag in memory.tags}:
                continue
            key, value = self._extract_fact_kv(memory.content)
            if not key or not value:
                continue
            grouped.setdefault(key, set()).add(value)
        return any(len(values) > 1 for values in grouped.values())

    def conflicting_keys(self, retrieved: list[RetrievedMemory]) -> list[str]:
        grouped: dict[str, set[str]] = {}
        for item in retrieved:
            memory = item.memory
            if memory.memory_type is not MemoryType.SEMANTIC:
                continue
            if "fact" not in {tag.lower() for tag in memory.tags}:
                continue
            key, value = self._extract_fact_kv(memory.content)
            if not key or not value:
                continue
            grouped.setdefault(key, set()).add(value)
        return [key for key, values in grouped.items() if len(values) > 1]

    def answer_memory_mismatch(self, answer: str, retrieved: list[RetrievedMemory]) -> bool:
        if not answer or not retrieved:
            return False

        answer_lower = answer.lower()
        memory_text = " ".join(item.memory.content.lower() for item in retrieved)
        anchors = [
            ("budget", ["budget", "예산"]),
            ("deadline", ["deadline", "마감"]),
            ("price", ["price", "가격"]),
            ("version", ["version", "버전"]),
        ]
        for _, alias_tokens in anchors:
            if any(token in answer_lower for token in alias_tokens) and not any(
                token in memory_text for token in alias_tokens
            ):
                return True

        return False

    def has_fact_support(self, retrieved: list[RetrievedMemory]) -> bool:
        return any("fact" in {tag.lower() for tag in item.memory.tags} for item in retrieved[:3])

    def has_preference_support(self, retrieved: list[RetrievedMemory]) -> bool:
        return any("preference" in {tag.lower() for tag in item.memory.tags} for item in retrieved[:3])

    def _verify_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievedMemory],
    ) -> tuple[str, list[str], float]:
        if claim.claim_type == "fact":
            return self._verify_fact_claim(claim, retrieved)
        if claim.claim_type == "preference":
            return self._verify_preference_claim(claim, retrieved)
        if claim.claim_type == "goal":
            return self._verify_goal_claim(claim, retrieved)
        return self._verify_generic_claim(claim, retrieved)

    def _verify_fact_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievedMemory],
    ) -> tuple[str, list[str], float]:
        fact_entries = self._fact_entries(retrieved)
        if claim.key:
            same_key = [item for item in fact_entries if item["key"] == claim.key]
            if not same_key:
                return "unsupported", [], 0.0
            if claim.value:
                exact_hits = [
                    item["memory_id"]
                    for item in same_key
                    if self._normalize_value(item["value"]) == self._normalize_value(claim.value)
                ]
                if exact_hits:
                    return "supported", exact_hits, 1.0
                return "contradicted", [item["memory_id"] for item in same_key], 0.9
            return "supported", [item["memory_id"] for item in same_key], 0.65

        if claim.value:
            value_hits = [
                item["memory_id"]
                for item in fact_entries
                if self._normalize_value(item["value"]) == self._normalize_value(claim.value)
            ]
            if value_hits:
                return "supported", value_hits, 0.60
            return "unsupported", [], 0.0

        return "unsupported", [], 0.0

    def _verify_preference_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievedMemory],
    ) -> tuple[str, list[str], float]:
        preference_memories = [
            item.memory
            for item in retrieved
            if "preference" in {tag.lower() for tag in item.memory.tags}
        ]
        if not preference_memories:
            return "unsupported", [], 0.0

        key = (claim.key or "style").lower().strip()
        value = (claim.value or "").lower().strip()
        support_hits: list[str] = []
        conflict_hits: list[str] = []

        for memory in preference_memories:
            attributes = self._preference_attributes(memory.content.lower())
            memory_value = attributes.get(key, "")
            if not memory_value and key != "style":
                continue
            if value in {"", "generic"}:
                if memory_value:
                    support_hits.append(memory.memory_id)
                continue
            if memory_value == value:
                support_hits.append(memory.memory_id)
            elif memory_value and memory_value != value:
                conflict_hits.append(memory.memory_id)

        if support_hits:
            return "supported", support_hits, 0.95
        if conflict_hits:
            return "contradicted", conflict_hits, 0.80
        return "unsupported", [], 0.0

    def _verify_goal_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievedMemory],
    ) -> tuple[str, list[str], float]:
        goal_memories = [
            item.memory
            for item in retrieved
            if "goal" in {tag.lower() for tag in item.memory.tags}
        ]
        if not goal_memories:
            return "unsupported", [], 0.0

        claim_tokens = self._content_tokens(claim.text)
        if not claim_tokens:
            return "unsupported", [], 0.0

        best_overlap = 0.0
        supported_ids: list[str] = []
        for memory in goal_memories:
            overlap = self._token_overlap_ratio(claim_tokens, self._content_tokens(memory.content))
            if overlap > best_overlap:
                best_overlap = overlap
            if overlap >= 0.30:
                supported_ids.append(memory.memory_id)

        if supported_ids:
            return "supported", supported_ids, best_overlap
        return "unsupported", [], 0.0

    def _verify_generic_claim(
        self,
        claim: Claim,
        retrieved: list[RetrievedMemory],
    ) -> tuple[str, list[str], float]:
        if not retrieved:
            return "unsupported", [], 0.0
        lowered = claim.text.lower().strip()
        if any(
            token in lowered
            for token in [
                "단정하기 어렵",
                "확실하지",
                "확인 필요",
                "추정하지",
                "uncertain",
                "not sure",
            ]
        ):
            return "supported", [retrieved[0].memory.memory_id], 0.55
        claim_tokens = self._content_tokens(claim.text)
        best_overlap = 0.0
        best_memory_id = ""
        for item in retrieved[:3]:
            overlap = self._token_overlap_ratio(claim_tokens, self._content_tokens(item.memory.content))
            if overlap > best_overlap:
                best_overlap = overlap
                best_memory_id = item.memory.memory_id
        if best_overlap >= 0.12 and best_memory_id:
            return "supported", [best_memory_id], best_overlap
        return "unsupported", [], 0.0

    def _preference_alignment_score(self, assessments: list[ClaimAssessment]) -> float:
        preference_assessments = [item for item in assessments if item.claim.claim_type == "preference"]
        if not preference_assessments:
            return 1.0
        supported = sum(1 for item in preference_assessments if item.status == "supported")
        return supported / len(preference_assessments)

    def _fact_entries(self, retrieved: list[RetrievedMemory]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for item in retrieved:
            memory = item.memory
            if memory.memory_type is not MemoryType.SEMANTIC:
                continue
            if "fact" not in {tag.lower() for tag in memory.tags}:
                continue
            key, value = self._extract_fact_kv(memory.content)
            if not key and not value:
                continue
            entries.append(
                {
                    "memory_id": memory.memory_id,
                    "key": key,
                    "value": value,
                    "content": memory.content,
                }
            )
        return entries

    def _extract_fact_kv(self, content: str) -> tuple[str, str]:
        lowered = content.lower()
        structured = re.search(r"key=([a-z_]+)\s*;\s*value=([^;]+)", lowered)
        if structured:
            return structured.group(1).strip(), self._normalize_value(structured.group(2))

        num = re.search(r"(\d+(?:\.\d+)?)", lowered)
        key = ""
        if "budget" in lowered or "예산" in lowered:
            key = "budget"
        elif "deadline" in lowered or "마감" in lowered:
            key = "deadline"
        elif "price" in lowered or "가격" in lowered:
            key = "price"
        elif "version" in lowered or "버전" in lowered:
            key = "version"
        elif "rate" in lowered or "비율" in lowered or "%" in lowered:
            key = "rate"
        if key and num:
            return key, num.group(1)
        return "", ""

    def _normalize_value(self, raw: str) -> str:
        value = raw.strip().lower()
        numeric = re.search(r"\d+(?:\.\d+)?", value)
        if numeric:
            return numeric.group(0)
        return value

    def _preference_attributes(self, text: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        if any(token in text for token in ["간결", "짧", "concise", "short"]):
            attrs["style"] = "concise"
        elif any(token in text for token in ["자세", "길게", "detailed", "long"]):
            attrs["style"] = "detailed"
        if any(token in text for token in ["bullet", "불릿", "목록"]):
            attrs["format"] = "bullet"
        elif any(token in text for token in ["문단", "paragraph", "불릿 말고", "no bullet"]):
            attrs["format"] = "no_bullet"
        if any(token in text for token in ["정중", "formal", "polite", "공손"]):
            attrs["tone"] = "formal"
        elif any(token in text for token in ["편하게", "casual", "친근"]):
            attrs["tone"] = "casual"
        return attrs

    def _content_tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9가-힣]+", text.lower())
        return {token for token in tokens if len(token) > 1}

    def _token_overlap_ratio(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)
