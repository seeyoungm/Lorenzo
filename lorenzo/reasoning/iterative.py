from __future__ import annotations

from dataclasses import dataclass

from lorenzo.models import ProcessedInput, ReasoningPlan, RetrievedMemory
from lorenzo.reasoning.claim_extractor import ClaimExtractor
from lorenzo.reasoning.refinement_judge import (
    ClaimAwareRefinementJudge,
    RefinementAcceptPolicy,
    VerificationSummary,
)
from lorenzo.reasoning.refinement_objectives import (
    RefinementObjectivePolicy,
    RefinementObjectiveRouter,
)
from lorenzo.reasoning.refinement_rewriter import ConservativeRefinementRewriter
from lorenzo.reasoning.requery_builder import RequeryBuilder


@dataclass(slots=True)
class IterativeRefinementResult:
    response: str
    plan: ReasoningPlan
    retrieved_memories: list[RetrievedMemory]
    iterations_used: int
    refinement_triggered: bool
    conflict_detected: bool
    answer_changed: bool
    iteration_gain: float
    trigger_reasons: list[str]
    draft_support_score: float
    final_support_score: float
    factual_refinement_attempted: bool
    factual_refinement_improved: bool
    preference_alignment_attempted: bool
    preference_alignment_improved: bool
    support_completion_attempted: bool
    support_completion_improved: bool
    conflict_fix_attempted: bool
    conflict_fix_succeeded: bool
    claim_support_coverage: float
    unsupported_claim_rate: float
    contradiction_reduced: bool
    refinement_regressed: bool
    retrieval_improved_but_answer_worsened: bool
    answer_changed_without_support_improvement: bool
    unsupported_claims_remaining: bool
    refinement_improved: bool
    draft_claim_support_coverage: float
    draft_unsupported_claim_rate: float


class IterativeReasoningEngine:
    def __init__(
        self,
        max_iterations: int = 2,
        low_confidence_threshold: float = 0.38,
        insufficient_similarity_threshold: float = 0.24,
        min_supporting_memories: int = 1,
        support_gain_margin: float = 0.03,
        min_support_coverage_gain: float = 0.03,
        min_preference_alignment_gain: float = 0.10,
        max_support_coverage_drop: float = 0.02,
    ) -> None:
        # Keep loop bounded to avoid unbounded reasoning cycles.
        self.max_iterations = max(2, min(3, max_iterations))
        self.support_gain_margin = max(0.0, support_gain_margin)

        objective_policy = RefinementObjectivePolicy(
            low_confidence_threshold=low_confidence_threshold,
            insufficient_similarity_threshold=insufficient_similarity_threshold,
            min_supporting_memories=max(1, min_supporting_memories),
        )
        judge_policy = RefinementAcceptPolicy(
            min_support_coverage_gain=max(0.0, min_support_coverage_gain),
            min_preference_alignment_gain=max(0.0, min_preference_alignment_gain),
            max_support_coverage_drop=max(0.0, max_support_coverage_drop),
        )

        self.claim_extractor = ClaimExtractor()
        self.objective_router = RefinementObjectiveRouter(policy=objective_policy)
        self.requery_builder = RequeryBuilder()
        self.judge = ClaimAwareRefinementJudge(policy=judge_policy)
        self.rewriter = ConservativeRefinementRewriter()

    def refine(
        self,
        *,
        user_input: str,
        processed: ProcessedInput,
        existing_memories,
        initial_retrieved: list[RetrievedMemory],
        draft_plan: ReasoningPlan,
        draft_response: str,
        memory_retriever,
        reasoning_planner,
        language_adapter,
        top_k: int,
    ) -> IterativeRefinementResult:
        current_response = draft_response
        current_plan = draft_plan
        current_retrieved = list(initial_retrieved)

        draft_support = self._support_score(initial_retrieved)
        current_support = draft_support

        draft_claims = self.claim_extractor.extract_claims(
            draft_answer=draft_response,
            user_input=user_input,
            processed=processed,
        )
        draft_summary = self.judge.verify_claims(draft_claims, initial_retrieved)
        current_summary = draft_summary

        refinement_triggered = False
        iterations_used = 1
        trigger_reasons: list[str] = []
        conflict_detected = self.judge.has_conflicting_memories(current_retrieved)
        factual_attempted = False
        factual_improved = False
        preference_attempted = False
        preference_improved = False
        support_attempted = False
        support_improved = False
        conflict_attempted = False
        conflict_fixed = False
        refinement_regressed = False
        retrieval_improved_but_answer_worsened = False
        refinement_improved = False

        for iteration in range(2, self.max_iterations + 1):
            iterations_used = iteration
            has_mismatch = self.judge.answer_memory_mismatch(current_response, current_retrieved)
            objectives = self.objective_router.derive(
                user_input=user_input,
                processed=processed,
                has_conflict=self.judge.has_conflicting_memories(current_retrieved),
                has_fact_support=self.judge.has_fact_support(current_retrieved),
                has_preference_support=self.judge.has_preference_support(current_retrieved),
                answer_memory_mismatch=has_mismatch,
                current_support=current_support,
                retrieved=current_retrieved,
            )
            objective_routes = self.objective_router.objective_routes(objectives)
            if not objective_routes:
                break

            refinement_triggered = True
            trigger_reasons.extend(objective_routes)
            if objectives.factual_correction:
                factual_attempted = True
            if objectives.preference_alignment:
                preference_attempted = True
            if objectives.support_completion:
                support_attempted = True
            if objectives.conflict_resolution:
                conflict_attempted = True

            unresolved_types = self.objective_router.prioritized_unresolved_types(objectives)
            claim_priority = self.objective_router.claim_priority_for_query(
                user_input=user_input,
                processed=processed,
            )
            requery = self.requery_builder.build(
                user_input=user_input,
                draft_answer=current_response,
                routes=objective_routes,
                unresolved_types=unresolved_types,
                claim_priority=claim_priority,
                claim_assessments=current_summary.assessments,
            )
            reretrieved = memory_retriever.retrieve(
                query=requery,
                memories=existing_memories,
                top_k=top_k,
                mode="with_fallback",
            )
            refined_plan = reasoning_planner.plan(processed, reretrieved)
            refined_response = language_adapter.generate(
                user_input=user_input,
                processed=processed,
                retrieved=reretrieved,
                plan=refined_plan,
            )
            refined_support = self._support_score(reretrieved)
            refined_claims = self.claim_extractor.extract_claims(
                draft_answer=refined_response,
                user_input=user_input,
                processed=processed,
            )
            refined_summary = self.judge.verify_claims(refined_claims, reretrieved)

            final_conflict = self.judge.has_conflicting_memories(reretrieved)
            final_mismatch = self.judge.answer_memory_mismatch(refined_response, reretrieved)
            conflict_detected = (
                conflict_detected
                or objectives.conflict_resolution
                or final_conflict
                or final_mismatch
            )

            decision = self.judge.judge_refinement(
                draft_summary=current_summary,
                refined_summary=refined_summary,
                draft_support_score=current_support,
                refined_support_score=refined_support,
                support_gain_margin=self.support_gain_margin,
            )

            # Reject risky updates: retrieval can improve while generated answer still mismatches evidence.
            if final_mismatch and decision.apply_refinement:
                decision.apply_refinement = False
                decision.regression = True
                decision.retrieval_improved_but_answer_worsened = True
                decision.reasons.append("final_answer_memory_mismatch")

            actionable_rewrite = self._has_actionable_claim_failures(refined_summary)
            if (not decision.apply_refinement) and decision.prefer_conservative_rewrite and actionable_rewrite:
                rewritten_response, rewritten_summary, rewritten_mismatch, rewrite_applied = (
                    self._attempt_conservative_rewrite(
                        user_input=user_input,
                        processed=processed,
                        draft_summary=current_summary,
                        refined_response=refined_response,
                        refined_summary=refined_summary,
                        reretrieved=reretrieved,
                        final_mismatch=final_mismatch,
                    )
                )
                if rewrite_applied:
                    refined_response = rewritten_response
                    refined_summary = rewritten_summary
                    final_mismatch = rewritten_mismatch
                    decision.apply_refinement = True
                    decision.improvement = True
                    decision.regression = False
                    decision.retrieval_improved_but_answer_worsened = False
                    decision.support_coverage_increased = (
                        rewritten_summary.support_coverage > current_summary.support_coverage
                    )
                    decision.unsupported_reduced = (
                        rewritten_summary.unsupported_count < current_summary.unsupported_count
                    )
                    decision.contradiction_reduced = (
                        rewritten_summary.contradicted_count < current_summary.contradicted_count
                    )
                    decision.evidence_strength_improved = (
                        rewritten_summary.avg_evidence_strength
                        > current_summary.avg_evidence_strength
                    )
                    decision.reasons.append("conservative_rewrite_applied")

            factual_step_improved = objectives.factual_correction and (
                decision.support_coverage_increased
                or decision.unsupported_reduced
                or decision.contradiction_reduced
                or decision.evidence_strength_improved
                or (
                    decision.apply_refinement
                    and self.judge.has_fact_support(reretrieved)
                    and (not final_mismatch)
                )
            )
            preference_step_improved = objectives.preference_alignment and (
                decision.preference_alignment_improved
                or (
                    decision.apply_refinement
                    and self.judge.has_preference_support(reretrieved)
                    and (not final_mismatch)
                )
                or decision.apply_refinement
            )
            support_step_improved = objectives.support_completion and (
                decision.support_coverage_increased
                or decision.unsupported_reduced
                or decision.evidence_strength_improved
                or (
                    decision.apply_refinement
                    and (
                        not self.objective_router.insufficient_supporting_memory(reretrieved)
                        or refined_support > (current_support + self.support_gain_margin)
                    )
                )
            )
            conflict_step_fixed = objectives.conflict_resolution and (
                (not final_conflict)
            )

            factual_improved = factual_improved or factual_step_improved
            preference_improved = preference_improved or preference_step_improved
            support_improved = support_improved or support_step_improved
            conflict_fixed = conflict_fixed or conflict_step_fixed
            refinement_regressed = refinement_regressed or (
                decision.apply_refinement and decision.regression
            )
            retrieval_improved_but_answer_worsened = (
                retrieval_improved_but_answer_worsened
                or decision.retrieval_improved_but_answer_worsened
            )

            if decision.apply_refinement:
                current_response = refined_response
                current_plan = refined_plan
                current_retrieved = reretrieved
                current_support = refined_support
                current_summary = refined_summary
                refinement_improved = refinement_improved or decision.improvement

            # v2 currently uses up to 2-pass by default.
            if self.max_iterations <= 2:
                break

        final_trigger_reasons = list(dict.fromkeys(trigger_reasons))
        answer_changed = current_response.strip() != draft_response.strip()
        quality_worsened = (
            current_summary.support_coverage + 1e-9 < draft_summary.support_coverage
            or current_summary.unsupported_count > draft_summary.unsupported_count
            or current_summary.contradicted_count > draft_summary.contradicted_count
        )
        return IterativeRefinementResult(
            response=current_response,
            plan=current_plan,
            retrieved_memories=current_retrieved,
            iterations_used=iterations_used,
            refinement_triggered=refinement_triggered,
            conflict_detected=conflict_detected,
            answer_changed=answer_changed,
            iteration_gain=current_support - draft_support,
            trigger_reasons=final_trigger_reasons,
            draft_support_score=draft_support,
            final_support_score=current_support,
            factual_refinement_attempted=factual_attempted,
            factual_refinement_improved=factual_improved,
            preference_alignment_attempted=preference_attempted,
            preference_alignment_improved=preference_improved,
            support_completion_attempted=support_attempted,
            support_completion_improved=support_improved,
            conflict_fix_attempted=conflict_attempted,
            conflict_fix_succeeded=conflict_fixed,
            claim_support_coverage=current_summary.support_coverage,
            unsupported_claim_rate=current_summary.unsupported_rate,
            contradiction_reduced=current_summary.contradicted_count < draft_summary.contradicted_count,
            refinement_regressed=refinement_regressed,
            retrieval_improved_but_answer_worsened=retrieval_improved_but_answer_worsened,
            answer_changed_without_support_improvement=(
                answer_changed
                and quality_worsened
            ),
            unsupported_claims_remaining=current_summary.unsupported_count > 0,
            refinement_improved=refinement_improved,
            draft_claim_support_coverage=draft_summary.support_coverage,
            draft_unsupported_claim_rate=draft_summary.unsupported_rate,
        )

    def _attempt_conservative_rewrite(
        self,
        *,
        user_input: str,
        processed: ProcessedInput,
        draft_summary: VerificationSummary,
        refined_response: str,
        refined_summary: VerificationSummary,
        reretrieved: list[RetrievedMemory],
        final_mismatch: bool,
    ) -> tuple[str, VerificationSummary, bool, bool]:
        rewritten = self.rewriter.rewrite(
            answer=refined_response,
            summary=refined_summary,
            retrieved=reretrieved,
            reason_flags={
                "unsupported_claims_remaining": refined_summary.unsupported_count > 0,
                "contradiction_persisted": refined_summary.contradicted_count > 0,
                "answer_memory_mismatch": final_mismatch,
            },
        )
        rewritten_claims = self.claim_extractor.extract_claims(
            draft_answer=rewritten,
            user_input=user_input,
            processed=processed,
        )
        rewritten_summary = self.judge.verify_claims(rewritten_claims, reretrieved)
        rewritten_mismatch = self.judge.answer_memory_mismatch(rewritten, reretrieved)

        rewrite_improved = (
            rewritten_summary.unsupported_count < refined_summary.unsupported_count
            or rewritten_summary.contradicted_count < refined_summary.contradicted_count
            or rewritten_summary.avg_evidence_strength > refined_summary.avg_evidence_strength
            or ((not rewritten_mismatch) and final_mismatch)
        )
        rewrite_safe = (
            rewritten_summary.unsupported_count <= draft_summary.unsupported_count
            and rewritten_summary.contradicted_count <= draft_summary.contradicted_count
            and (not rewritten_mismatch)
        )
        apply = rewrite_improved or rewrite_safe
        return rewritten, rewritten_summary, rewritten_mismatch, apply

    def _has_actionable_claim_failures(self, summary: VerificationSummary) -> bool:
        for assessment in summary.assessments:
            if assessment.claim.claim_type not in {"fact", "preference", "goal"}:
                continue
            if assessment.status in {"unsupported", "contradicted"}:
                return True
        return False

    def _support_score(self, retrieved: list[RetrievedMemory]) -> float:
        if not retrieved:
            return 0.0
        top = retrieved[:2]
        scores: list[float] = []
        for item in top:
            score = (
                (item.similarity_score * 0.65)
                + (item.importance_score * 0.20)
                + (item.recency_score * 0.15)
            )
            scores.append(score)
        return sum(scores) / len(scores)
