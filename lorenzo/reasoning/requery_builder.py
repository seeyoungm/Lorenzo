from __future__ import annotations

from lorenzo.reasoning.refinement_judge import ClaimAssessment


class RequeryBuilder:
    def build(
        self,
        *,
        user_input: str,
        draft_answer: str,
        routes: list[str],
        unresolved_types: list[str],
        claim_priority: list[str],
        claim_assessments: list[ClaimAssessment],
    ) -> str:
        routes_text = ", ".join(routes) if routes else "none"
        unresolved_text = ", ".join(unresolved_types) if unresolved_types else "general"
        priority_text = ", ".join(claim_priority) if claim_priority else "fact, preference, goal"

        prioritized_claims = self._prioritize_claims(
            claim_assessments=claim_assessments,
            claim_priority=claim_priority,
        )
        claims_text = "\n".join(prioritized_claims) if prioritized_claims else "- none"

        return (
            f"Original query:\n{user_input}\n\n"
            f"Draft answer for refinement:\n{draft_answer}\n\n"
            f"Refinement routes: {routes_text}\n"
            f"Prioritize unresolved memory types: {unresolved_text}\n"
            f"Claim priority by intent: {priority_text}\n"
            f"Claim verification results:\n{claims_text}\n\n"
            "Re-retrieve memories to verify contradicted/unsupported claims first, then fill missing support."
        )

    def _prioritize_claims(
        self,
        *,
        claim_assessments: list[ClaimAssessment],
        claim_priority: list[str],
    ) -> list[str]:
        if not claim_assessments:
            return []

        priority_rank = {claim_type: rank for rank, claim_type in enumerate(claim_priority)}
        # Focus pass-2 on unresolved claim buckets first.
        unresolved = [
            item
            for item in claim_assessments
            if item.status in {"contradicted", "unsupported"}
        ]
        if not unresolved:
            unresolved = claim_assessments[:]

        unresolved.sort(
            key=lambda item: (
                priority_rank.get(item.claim.claim_type, 99),
                0 if item.status == "contradicted" else 1,
                item.claim.key,
                item.claim.value,
                item.claim.text,
            )
        )

        rows: list[str] = []
        for item in unresolved[:8]:
            claim = item.claim
            if claim.key and claim.value:
                descriptor = f"verify key={claim.key}; value={claim.value}"
            elif claim.key:
                descriptor = f"verify key={claim.key}"
            else:
                descriptor = f"verify {claim.claim_type} claim={claim.text[:100]}"
            rows.append(f"- {item.status} -> {descriptor}")
        return rows
