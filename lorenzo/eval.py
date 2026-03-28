from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from lorenzo.config import AppConfig, load_config
from lorenzo.input_processor import InputProcessor
from lorenzo.language.backends import RuleBasedBackend
from lorenzo.models import MemoryItem, MemoryType, RetrievedMemory
from lorenzo.orchestrator import LorenzoOrchestrator
from lorenzo.reasoning import ReasoningPlanner

EVAL_TYPES = ("goal", "fact", "preference", "commitment", "event")


@dataclass(slots=True)
class EvalScenario:
    scenario_id: str
    category: str
    user_input: str
    expected_retrieval_keywords: list[str]
    should_store: bool | None
    expected_store_type: str | None
    expected_conflict_winner_keywords: list[str]
    stale_conflict_keywords: list[str]
    session_id: str
    turn: int
    consistency_group: str | None
    expected_goal_confidence: str | None = None
    goal_false_source: str | None = None
    goal_intrusion_probe: bool = False


@dataclass(slots=True)
class EvalMetrics:
    retrieval_hit_rate_top1: float
    retrieval_hit_rate_top1_strong_only: float
    retrieval_hit_rate_top1_with_fallback: float
    retrieval_hit_rate_top3: float
    response_consistency: float
    memory_precision: float
    memory_recall: float
    memory_recall_goal: float
    memory_recall_preference: float
    memory_recall_fact: float
    memory_growth_per_turn: float
    memory_growth_stability: float
    retrieval_degradation_over_time: float
    merge_activation_rate: float
    false_merge_rate: float
    merge_false_reject_rate: float
    merge_candidate_similarity_avg: float
    merge_rejected_reason: dict[str, int]
    rejected_count_by_type: dict[str, int]
    merge_success_count: int
    merge_rejected_count: int
    conflict_resolution_count: int
    conflict_resolution_accuracy: float
    stale_memory_usage_rate: float
    recall_by_type: dict[str, float]
    precision_by_type: dict[str, float]
    storage_rate_by_type: dict[str, float]
    conflict_rate_by_type: dict[str, float]
    retrieval_top1_over_time: list[float]
    conflict_accumulation_rate: float
    goal_precision_strong: float
    goal_recall_strong: float
    weak_goal_rate: float
    false_goal_from_wish_rate: float
    false_goal_from_opinion_rate: float
    false_goal_from_temporary_desire_rate: float
    goal_intrusion_rate_in_retrieval_top1: float
    weak_memory_usage_rate: float
    weak_memory_promotion_rate: float
    false_positive_reintroduced_rate: float
    goal_recall_recovery_rate: float
    fallback_trigger_rate: float
    fallback_help_rate: float
    fallback_no_effect_rate: float
    fallback_harm_rate: float
    avg_rank_change_from_fallback: float
    weak_candidate_present_rate: float
    weak_candidate_selected_rate: float
    weak_coverage_rate: float
    weak_coverage_goal_recall: float
    weak_coverage_preference_recall: float
    weak_coverage_fact_recall: float
    weak_override_trigger_rate: float
    weak_override_candidate_rate: float
    weak_override_blocked_by_low_score_count: int
    weak_override_blocked_by_similarity_count: int
    weak_override_blocked_by_type_alignment_count: int
    weak_override_success_count: int


class BaselineEngine:
    """Baseline without memory retrieval/update."""

    def __init__(self) -> None:
        self.input_processor = InputProcessor()
        self.reasoning = ReasoningPlanner()
        self.language = RuleBasedBackend()

    def run_turn(self, text: str) -> tuple[str, str]:
        processed = self.input_processor.process(text)
        plan = self.reasoning.plan(processed, [])
        response = self.language.generate(
            user_input=text,
            processed=processed,
            retrieved=[],
            plan=plan,
        )
        return response, plan.strategy


def load_scenarios(path: str | Path) -> list[EvalScenario]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios: list[EvalScenario] = []
    for idx, item in enumerate(raw, start=1):
        scenarios.append(
            EvalScenario(
                scenario_id=str(item.get("id", f"scenario-{idx}")),
                category=item.get("category", "uncategorized"),
                user_input=item["user_input"],
                expected_retrieval_keywords=list(item.get("expected_retrieval_keywords", [])),
                should_store=item.get("should_store"),
                expected_store_type=item.get("expected_store_type"),
                expected_conflict_winner_keywords=list(item.get("expected_conflict_winner_keywords", [])),
                stale_conflict_keywords=list(item.get("stale_conflict_keywords", [])),
                session_id=item.get("session_id", "default"),
                turn=int(item.get("turn", idx)),
                consistency_group=item.get("consistency_group"),
                expected_goal_confidence=item.get("expected_goal_confidence"),
                goal_false_source=item.get("goal_false_source"),
                goal_intrusion_probe=bool(item.get("goal_intrusion_probe", False)),
            )
        )

    # Preserve session turn order.
    return sorted(scenarios, key=lambda s: (s.session_id, s.turn, s.scenario_id))


def evaluate_memory_pipeline(config: AppConfig, scenarios: list[EvalScenario]) -> EvalMetrics:
    retrieval_hits_top1 = 0
    retrieval_hits_top1_strong_only = 0
    retrieval_hits_top1_with_fallback = 0
    retrieval_hits_top3 = 0
    retrieval_total = 0

    tp_store = 0
    fp_store = 0
    total_growth = 0
    growth_history: list[int] = []
    total_turns = len(scenarios)

    expected_store_total = 0
    stored_expected_total = 0
    expected_by_type: dict[str, int] = {t: 0 for t in EVAL_TYPES}
    stored_by_type: dict[str, int] = {t: 0 for t in EVAL_TYPES}
    fp_by_type: dict[str, int] = {t: 0 for t in EVAL_TYPES}
    actual_stored_count_by_type: dict[str, int] = {t: 0 for t in EVAL_TYPES}
    conflict_count_by_type: dict[str, int] = {t: 0 for t in EVAL_TYPES}

    group_to_strategies: dict[str, list[str]] = {}
    merge_attempts = 0
    merge_applied = 0
    merge_success_count = 0
    merge_rejected_count = 0
    merge_false_reject_count = 0
    merge_rejected_reason: dict[str, int] = {}
    rejected_count_by_type: dict[str, int] = {}
    merge_candidate_similarity_values: list[float] = []
    false_merge_attempts = 0
    conflict_resolution_count = 0
    candidates_seen = 0
    weak_memory_stored_count = 0
    weak_memory_promotion_count = 0

    conflict_checks = 0
    conflict_correct = 0
    stale_top1_count = 0
    goal_expected_strong = 0
    goal_predicted_strong = 0
    goal_true_positive_strong = 0
    goal_labeled_total = 0
    weak_goal_count = 0
    false_goal_source_total = {"wish": 0, "opinion": 0, "temporary_desire": 0}
    false_goal_source_strong = {"wish": 0, "opinion": 0, "temporary_desire": 0}
    goal_intrusion_probes = 0
    goal_intrusion_count = 0
    weak_memory_usage_turns = 0
    false_positive_reintroduced = 0
    strong_top1_hit_total = 0
    goal_recall_strong_miss_total = 0
    goal_recall_recovered_total = 0
    fallback_trigger_count = 0
    fallback_help_count = 0
    fallback_no_effect_count = 0
    fallback_harm_count = 0
    fallback_rank_changes: list[int] = []
    weak_candidate_present_count = 0
    weak_candidate_selected_count = 0
    weak_override_trigger_count = 0
    weak_override_candidate_turn_count = 0
    weak_override_blocked_by_low_score_count = 0
    weak_override_blocked_by_similarity_count = 0
    weak_override_blocked_by_type_alignment_count = 0
    weak_override_success_count = 0
    recall_query_total_by_intent = {"goal_recall": 0, "preference_recall": 0, "fact_recall": 0}
    weak_coverage_hits_by_intent = {"goal_recall": 0, "preference_recall": 0, "fact_recall": 0}

    long_top1_by_session: dict[str, list[tuple[int, bool]]] = {}
    long_top1_by_turn: dict[int, list[int]] = {}
    session_max_turn: dict[str, int] = {}
    for scenario in scenarios:
        if scenario.category == "long_session":
            session_max_turn[scenario.session_id] = max(
                session_max_turn.get(scenario.session_id, 0), scenario.turn
            )

    session_ctx: dict[str, tuple[tempfile.TemporaryDirectory[str], LorenzoOrchestrator]] = {}
    try:
        for scenario in scenarios:
            if scenario.session_id not in session_ctx:
                temp_dir = tempfile.TemporaryDirectory(prefix=f"lorenzo-eval-{scenario.session_id}-")
                temp_path = Path(temp_dir.name) / "memory_eval.jsonl"
                if config.memory.path.exists():
                    shutil.copy(config.memory.path, temp_path)
                else:
                    temp_path.touch()

                session_cfg = AppConfig(
                    memory=replace(config.memory, path=temp_path),
                    language_backend=config.language_backend,
                )
                session_ctx[scenario.session_id] = (
                    temp_dir,
                    LorenzoOrchestrator.from_config(session_cfg),
                )

            _, orchestrator = session_ctx[scenario.session_id]

            processed_for_eval = orchestrator.modules.input_processor.process(scenario.user_input)
            predicted_goal_confidence = processed_for_eval.goal_confidence
            expected_goal_confidence = (
                scenario.expected_goal_confidence.lower().strip()
                if scenario.expected_goal_confidence
                else None
            )
            if expected_goal_confidence in {"strong", "weak", "none"}:
                goal_labeled_total += 1
                if expected_goal_confidence == "strong":
                    goal_expected_strong += 1
                if predicted_goal_confidence == "strong":
                    goal_predicted_strong += 1
                    if expected_goal_confidence == "strong":
                        goal_true_positive_strong += 1
                if predicted_goal_confidence == "weak":
                    weak_goal_count += 1

            false_goal_source = (
                scenario.goal_false_source.lower().strip()
                if scenario.goal_false_source
                else None
            )
            if false_goal_source in false_goal_source_total:
                false_goal_source_total[false_goal_source] += 1
                if predicted_goal_confidence == "strong":
                    false_goal_source_strong[false_goal_source] += 1

            before_telemetry = orchestrator.snapshot_telemetry()
            before_memories = orchestrator.modules.memory_store.list_all()
            strong_only_ranked = orchestrator.modules.memory_retriever.retrieve(
                query=scenario.user_input,
                memories=before_memories,
                top_k=orchestrator.top_k,
                mode="strong_only",
            )
            fallback_ranked = orchestrator.modules.memory_retriever.retrieve(
                query=scenario.user_input,
                memories=before_memories,
                top_k=orchestrator.top_k,
                mode="with_fallback",
            )
            fallback_triggered = bool(
                getattr(orchestrator.modules.memory_retriever, "last_fallback_triggered", False)
            )
            weak_override_triggered = bool(
                getattr(orchestrator.modules.memory_retriever, "last_weak_override_triggered", False)
            )
            weak_override_candidates_this_turn = int(
                getattr(
                    orchestrator.modules.memory_retriever,
                    "last_weak_override_candidate_count",
                    0,
                )
            )
            weak_override_blocked_low_this_turn = int(
                getattr(
                    orchestrator.modules.memory_retriever,
                    "last_weak_override_blocked_by_low_score_count",
                    0,
                )
            )
            weak_override_blocked_sim_this_turn = int(
                getattr(
                    orchestrator.modules.memory_retriever,
                    "last_weak_override_blocked_by_similarity_count",
                    0,
                )
            )
            weak_override_blocked_type_this_turn = int(
                getattr(
                    orchestrator.modules.memory_retriever,
                    "last_weak_override_blocked_by_type_alignment_count",
                    0,
                )
            )
            weak_override_success_this_turn = int(
                getattr(
                    orchestrator.modules.memory_retriever,
                    "last_weak_override_success_count",
                    0,
                )
            )
            strong_top1_text = (
                strong_only_ranked[0].memory.content.lower() if strong_only_ranked else ""
            )
            fallback_top1_text = (
                fallback_ranked[0].memory.content.lower() if fallback_ranked else ""
            )
            weak_candidate_present = any(_is_weak_memory(item) for item in before_memories)
            weak_candidate_selected = bool(fallback_ranked) and _is_weak_memory(fallback_ranked[0].memory)
            before_count = len(before_memories)
            result = orchestrator.run_turn(scenario.user_input)
            after_count = orchestrator.modules.memory_store.count()
            after_telemetry = orchestrator.snapshot_telemetry()
            after_memories = orchestrator.modules.memory_store.list_all()
            before_ids = {item.memory_id for item in before_memories}
            added_memories = [item for item in after_memories if item.memory_id not in before_ids]

            growth = max(0, after_count - before_count)
            total_growth += growth
            growth_history.append(growth)

            merge_attempts += (after_telemetry.merge_attempts - before_telemetry.merge_attempts)
            merge_applied += (after_telemetry.merge_applied - before_telemetry.merge_applied)
            merge_success_count += (
                after_telemetry.merge_success_count - before_telemetry.merge_success_count
            )
            merge_rejected_count += (
                after_telemetry.merge_rejected_count - before_telemetry.merge_rejected_count
            )
            merge_false_reject_count += (
                after_telemetry.merge_false_reject_count - before_telemetry.merge_false_reject_count
            )
            for key, value in after_telemetry.merge_rejected_reason.items():
                delta = value - before_telemetry.merge_rejected_reason.get(key, 0)
                if delta > 0:
                    merge_rejected_reason[key] = merge_rejected_reason.get(key, 0) + delta
            for key, value in after_telemetry.rejected_count_by_type.items():
                delta = value - before_telemetry.rejected_count_by_type.get(key, 0)
                if delta > 0:
                    rejected_count_by_type[key] = rejected_count_by_type.get(key, 0) + delta
            before_sim_count = len(before_telemetry.merge_candidate_similarity)
            if len(after_telemetry.merge_candidate_similarity) > before_sim_count:
                merge_candidate_similarity_values.extend(
                    after_telemetry.merge_candidate_similarity[before_sim_count:]
                )
            false_merge_attempts += (
                after_telemetry.false_merge_attempts - before_telemetry.false_merge_attempts
            )
            conflict_resolution_count += (
                after_telemetry.conflict_resolved - before_telemetry.conflict_resolved
            )
            for key, value in after_telemetry.conflict_count_by_type.items():
                delta = value - before_telemetry.conflict_count_by_type.get(key, 0)
                if delta > 0 and key in conflict_count_by_type:
                    conflict_count_by_type[key] += delta
            candidates_seen += (after_telemetry.candidates_seen - before_telemetry.candidates_seen)
            weak_memory_stored_count += (
                after_telemetry.weak_memory_stored - before_telemetry.weak_memory_stored
            )
            weak_memory_promotion_count += (
                after_telemetry.weak_memory_promotions - before_telemetry.weak_memory_promotions
            )

            top1_text = (result.retrieved_memories[0].memory.content if result.retrieved_memories else "").lower()
            top3_text = " ".join(item.memory.content for item in result.retrieved_memories[:3]).lower()
            top1_type = (
                _memory_eval_type(result.retrieved_memories[0].memory)
                if result.retrieved_memories
                else None
            )
            if any(_is_weak_memory(item.memory) for item in result.retrieved_memories):
                weak_memory_usage_turns += 1

            if scenario.expected_retrieval_keywords:
                retrieval_total += 1
                top1_hit = _contains_any_keyword(top1_text, scenario.expected_retrieval_keywords)
                strong_top1_hit = _contains_any_keyword(strong_top1_text, scenario.expected_retrieval_keywords)
                fallback_top1_hit = _contains_any_keyword(
                    fallback_top1_text,
                    scenario.expected_retrieval_keywords,
                )
                strong_best_rank = _best_hit_rank(
                    strong_only_ranked,
                    scenario.expected_retrieval_keywords,
                    top_k=orchestrator.top_k,
                )
                fallback_best_rank = _best_hit_rank(
                    fallback_ranked,
                    scenario.expected_retrieval_keywords,
                    top_k=orchestrator.top_k,
                )
                fallback_rank_changes.append(strong_best_rank - fallback_best_rank)
                top3_hit = _contains_any_keyword(top3_text, scenario.expected_retrieval_keywords)
                if top1_hit:
                    retrieval_hits_top1 += 1
                if strong_top1_hit:
                    retrieval_hits_top1_strong_only += 1
                    strong_top1_hit_total += 1
                if fallback_top1_hit:
                    retrieval_hits_top1_with_fallback += 1
                if top3_hit:
                    retrieval_hits_top3 += 1
                if strong_top1_hit and not fallback_top1_hit:
                    false_positive_reintroduced += 1
                if _is_goal_like_query(scenario.user_input):
                    if not strong_top1_hit:
                        goal_recall_strong_miss_total += 1
                        if fallback_top1_hit:
                            goal_recall_recovered_total += 1
                if weak_candidate_present:
                    weak_candidate_present_count += 1
                if weak_candidate_selected:
                    weak_candidate_selected_count += 1
                query_intent = orchestrator.modules.memory_retriever._infer_intent(scenario.user_input)
                if query_intent in recall_query_total_by_intent:
                    recall_query_total_by_intent[query_intent] += 1
                    aligned_weak_present = any(
                        _is_weak_memory(item)
                        and _weak_matches_intent_for_eval(item, query_intent)
                        for item in before_memories
                    )
                    if aligned_weak_present:
                        weak_coverage_hits_by_intent[query_intent] += 1
                if fallback_triggered:
                    fallback_trigger_count += 1
                    if (not strong_top1_hit) and fallback_top1_hit:
                        fallback_help_count += 1
                    elif strong_top1_hit and (not fallback_top1_hit):
                        fallback_harm_count += 1
                    else:
                        fallback_no_effect_count += 1
                if weak_override_candidates_this_turn > 0:
                    weak_override_candidate_turn_count += 1
                weak_override_blocked_by_low_score_count += weak_override_blocked_low_this_turn
                weak_override_blocked_by_similarity_count += weak_override_blocked_sim_this_turn
                weak_override_blocked_by_type_alignment_count += weak_override_blocked_type_this_turn
                weak_override_success_count += weak_override_success_this_turn
                if weak_override_triggered:
                    weak_override_trigger_count += 1

                if scenario.category == "long_session":
                    long_top1_by_session.setdefault(scenario.session_id, []).append((scenario.turn, top1_hit))
                    long_top1_by_turn.setdefault(scenario.turn, []).append(1 if top1_hit else 0)

            if scenario.should_store is not None:
                grew = growth > 0
                if grew and scenario.should_store:
                    tp_store += 1
                elif grew and not scenario.should_store:
                    fp_store += 1

            if scenario.should_store and scenario.expected_store_type:
                expected_store_total += 1
                expected_type = scenario.expected_store_type.lower().strip()
                if expected_type in expected_by_type:
                    expected_by_type[expected_type] += 1
                if _stored_expected_memory(
                    orchestrator.modules.memory_retriever,
                    after_memories,
                    scenario.user_input,
                    scenario.expected_store_type,
                ):
                    stored_expected_total += 1
                    if expected_type in stored_by_type:
                        stored_by_type[expected_type] += 1

            added_types_this_turn: set[str] = set()
            for item in added_memories:
                eval_type = _memory_eval_type(item)
                if eval_type is None:
                    continue
                actual_stored_count_by_type[eval_type] += 1
                added_types_this_turn.add(eval_type)

            if scenario.should_store is False:
                for t in added_types_this_turn:
                    fp_by_type[t] += 1
            elif scenario.should_store and scenario.expected_store_type:
                expected_type = scenario.expected_store_type.lower().strip()
                for t in added_types_this_turn:
                    if t != expected_type:
                        fp_by_type[t] += 1

            if scenario.expected_conflict_winner_keywords:
                conflict_checks += 1
                winner_hit = _contains_any_keyword(top1_text, scenario.expected_conflict_winner_keywords)
                if winner_hit:
                    conflict_correct += 1
                if scenario.stale_conflict_keywords and _contains_any_keyword(
                    top1_text, scenario.stale_conflict_keywords
                ) and not winner_hit:
                    stale_top1_count += 1

            if scenario.goal_intrusion_probe:
                goal_intrusion_probes += 1
                if top1_type == "goal":
                    goal_intrusion_count += 1

            if scenario.consistency_group:
                group_to_strategies.setdefault(scenario.consistency_group, []).append(result.plan.strategy)
    finally:
        for temp_dir, _ in session_ctx.values():
            temp_dir.cleanup()

    retrieval_hit_rate_top1 = _safe_div(retrieval_hits_top1, retrieval_total)
    retrieval_hit_rate_top1_strong_only = _safe_div(retrieval_hits_top1_strong_only, retrieval_total)
    retrieval_hit_rate_top1_with_fallback = _safe_div(retrieval_hits_top1_with_fallback, retrieval_total)
    retrieval_hit_rate_top3 = _safe_div(retrieval_hits_top3, retrieval_total)
    response_consistency = _group_consistency(group_to_strategies)
    memory_precision = _safe_div(tp_store, tp_store + fp_store) if (tp_store + fp_store) > 0 else 1.0
    memory_recall = _safe_div(stored_expected_total, expected_store_total)
    memory_recall_goal = _safe_div(stored_by_type["goal"], expected_by_type["goal"])
    memory_recall_preference = _safe_div(stored_by_type["preference"], expected_by_type["preference"])
    memory_recall_fact = _safe_div(stored_by_type["fact"], expected_by_type["fact"])
    recall_by_type = {t: _safe_div(stored_by_type[t], expected_by_type[t]) for t in EVAL_TYPES}
    precision_by_type = {
        t: _safe_div(stored_by_type[t], stored_by_type[t] + fp_by_type[t]) for t in EVAL_TYPES
    }
    storage_rate_by_type = {t: _safe_div(actual_stored_count_by_type[t], total_turns) for t in EVAL_TYPES}
    conflict_rate_by_type = {
        t: _safe_div(conflict_count_by_type[t], expected_by_type[t]) for t in EVAL_TYPES
    }
    memory_growth_per_turn = _safe_div(total_growth, total_turns)
    memory_growth_stability = _growth_stability(growth_history)
    retrieval_top1_over_time = _retrieval_top1_over_time(long_top1_by_turn)
    retrieval_degradation_over_time = _retrieval_degradation(long_top1_by_session, session_max_turn)
    merge_activation_rate = _safe_div(merge_applied, candidates_seen)
    false_merge_rate = _safe_div(false_merge_attempts, merge_attempts)
    merge_false_reject_rate = _safe_div(merge_false_reject_count, merge_rejected_count)
    merge_candidate_similarity_avg = (
        sum(merge_candidate_similarity_values) / len(merge_candidate_similarity_values)
        if merge_candidate_similarity_values
        else 0.0
    )
    conflict_resolution_accuracy = _safe_div(conflict_correct, conflict_checks)
    stale_memory_usage_rate = _safe_div(stale_top1_count, conflict_checks)
    conflict_accumulation_rate = _safe_div(conflict_resolution_count, total_turns)
    goal_precision_strong = _safe_div(goal_true_positive_strong, goal_predicted_strong)
    goal_recall_strong = _safe_div(goal_true_positive_strong, goal_expected_strong)
    weak_goal_rate = _safe_div(weak_goal_count, goal_labeled_total)
    false_goal_from_wish_rate = _safe_div(
        false_goal_source_strong["wish"], false_goal_source_total["wish"]
    )
    false_goal_from_opinion_rate = _safe_div(
        false_goal_source_strong["opinion"], false_goal_source_total["opinion"]
    )
    false_goal_from_temporary_desire_rate = _safe_div(
        false_goal_source_strong["temporary_desire"],
        false_goal_source_total["temporary_desire"],
    )
    goal_intrusion_rate_in_retrieval_top1 = _safe_div(goal_intrusion_count, goal_intrusion_probes)
    weak_memory_usage_rate = _safe_div(weak_memory_usage_turns, total_turns)
    weak_memory_promotion_rate = _safe_div(weak_memory_promotion_count, weak_memory_stored_count)
    false_positive_reintroduced_rate = _safe_div(false_positive_reintroduced, strong_top1_hit_total)
    goal_recall_recovery_rate = _safe_div(goal_recall_recovered_total, goal_recall_strong_miss_total)
    fallback_trigger_rate = _safe_div(fallback_trigger_count, retrieval_total)
    fallback_help_rate = _safe_div(fallback_help_count, fallback_trigger_count)
    fallback_no_effect_rate = _safe_div(fallback_no_effect_count, fallback_trigger_count)
    fallback_harm_rate = _safe_div(fallback_harm_count, fallback_trigger_count)
    avg_rank_change_from_fallback = (
        sum(fallback_rank_changes) / len(fallback_rank_changes) if fallback_rank_changes else 0.0
    )
    weak_candidate_present_rate = _safe_div(weak_candidate_present_count, retrieval_total)
    weak_candidate_selected_rate = _safe_div(weak_candidate_selected_count, weak_candidate_present_count)
    weak_coverage_total_hits = sum(weak_coverage_hits_by_intent.values())
    weak_coverage_total_turns = sum(recall_query_total_by_intent.values())
    weak_coverage_rate = _safe_div(weak_coverage_total_hits, weak_coverage_total_turns)
    weak_coverage_goal_recall = _safe_div(
        weak_coverage_hits_by_intent["goal_recall"],
        recall_query_total_by_intent["goal_recall"],
    )
    weak_coverage_preference_recall = _safe_div(
        weak_coverage_hits_by_intent["preference_recall"],
        recall_query_total_by_intent["preference_recall"],
    )
    weak_coverage_fact_recall = _safe_div(
        weak_coverage_hits_by_intent["fact_recall"],
        recall_query_total_by_intent["fact_recall"],
    )
    weak_override_trigger_rate = _safe_div(weak_override_trigger_count, retrieval_total)
    weak_override_candidate_rate = _safe_div(weak_override_candidate_turn_count, retrieval_total)

    return EvalMetrics(
        retrieval_hit_rate_top1=retrieval_hit_rate_top1,
        retrieval_hit_rate_top1_strong_only=retrieval_hit_rate_top1_strong_only,
        retrieval_hit_rate_top1_with_fallback=retrieval_hit_rate_top1_with_fallback,
        retrieval_hit_rate_top3=retrieval_hit_rate_top3,
        response_consistency=response_consistency,
        memory_precision=memory_precision,
        memory_recall=memory_recall,
        memory_recall_goal=memory_recall_goal,
        memory_recall_preference=memory_recall_preference,
        memory_recall_fact=memory_recall_fact,
        memory_growth_per_turn=memory_growth_per_turn,
        memory_growth_stability=memory_growth_stability,
        retrieval_degradation_over_time=retrieval_degradation_over_time,
        merge_activation_rate=merge_activation_rate,
        false_merge_rate=false_merge_rate,
        merge_false_reject_rate=merge_false_reject_rate,
        merge_candidate_similarity_avg=merge_candidate_similarity_avg,
        merge_rejected_reason=dict(sorted(merge_rejected_reason.items())),
        rejected_count_by_type=dict(sorted(rejected_count_by_type.items())),
        merge_success_count=merge_success_count,
        merge_rejected_count=merge_rejected_count,
        conflict_resolution_count=conflict_resolution_count,
        conflict_resolution_accuracy=conflict_resolution_accuracy,
        stale_memory_usage_rate=stale_memory_usage_rate,
        recall_by_type={k: round(v, 3) for k, v in sorted(recall_by_type.items())},
        precision_by_type={k: round(v, 3) for k, v in sorted(precision_by_type.items())},
        storage_rate_by_type={k: round(v, 3) for k, v in sorted(storage_rate_by_type.items())},
        conflict_rate_by_type={k: round(v, 3) for k, v in sorted(conflict_rate_by_type.items())},
        retrieval_top1_over_time=[round(v, 3) for v in retrieval_top1_over_time],
        conflict_accumulation_rate=conflict_accumulation_rate,
        goal_precision_strong=goal_precision_strong,
        goal_recall_strong=goal_recall_strong,
        weak_goal_rate=weak_goal_rate,
        false_goal_from_wish_rate=false_goal_from_wish_rate,
        false_goal_from_opinion_rate=false_goal_from_opinion_rate,
        false_goal_from_temporary_desire_rate=false_goal_from_temporary_desire_rate,
        goal_intrusion_rate_in_retrieval_top1=goal_intrusion_rate_in_retrieval_top1,
        weak_memory_usage_rate=weak_memory_usage_rate,
        weak_memory_promotion_rate=weak_memory_promotion_rate,
        false_positive_reintroduced_rate=false_positive_reintroduced_rate,
        goal_recall_recovery_rate=goal_recall_recovery_rate,
        fallback_trigger_rate=fallback_trigger_rate,
        fallback_help_rate=fallback_help_rate,
        fallback_no_effect_rate=fallback_no_effect_rate,
        fallback_harm_rate=fallback_harm_rate,
        avg_rank_change_from_fallback=avg_rank_change_from_fallback,
        weak_candidate_present_rate=weak_candidate_present_rate,
        weak_candidate_selected_rate=weak_candidate_selected_rate,
        weak_coverage_rate=weak_coverage_rate,
        weak_coverage_goal_recall=weak_coverage_goal_recall,
        weak_coverage_preference_recall=weak_coverage_preference_recall,
        weak_coverage_fact_recall=weak_coverage_fact_recall,
        weak_override_trigger_rate=weak_override_trigger_rate,
        weak_override_candidate_rate=weak_override_candidate_rate,
        weak_override_blocked_by_low_score_count=weak_override_blocked_by_low_score_count,
        weak_override_blocked_by_similarity_count=weak_override_blocked_by_similarity_count,
        weak_override_blocked_by_type_alignment_count=weak_override_blocked_by_type_alignment_count,
        weak_override_success_count=weak_override_success_count,
    )


def evaluate_baseline(scenarios: list[EvalScenario]) -> EvalMetrics:
    engine = BaselineEngine()
    group_to_strategies: dict[str, list[str]] = {}

    for scenario in scenarios:
        _, strategy = engine.run_turn(scenario.user_input)
        if scenario.consistency_group:
            group_to_strategies.setdefault(scenario.consistency_group, []).append(strategy)

    return EvalMetrics(
        retrieval_hit_rate_top1=0.0,
        retrieval_hit_rate_top1_strong_only=0.0,
        retrieval_hit_rate_top1_with_fallback=0.0,
        retrieval_hit_rate_top3=0.0,
        response_consistency=_group_consistency(group_to_strategies),
        memory_precision=1.0,
        memory_recall=0.0,
        memory_recall_goal=0.0,
        memory_recall_preference=0.0,
        memory_recall_fact=0.0,
        memory_growth_per_turn=0.0,
        memory_growth_stability=1.0,
        retrieval_degradation_over_time=0.0,
        merge_activation_rate=0.0,
        false_merge_rate=0.0,
        merge_false_reject_rate=0.0,
        merge_candidate_similarity_avg=0.0,
        merge_rejected_reason={},
        rejected_count_by_type={},
        merge_success_count=0,
        merge_rejected_count=0,
        conflict_resolution_count=0,
        conflict_resolution_accuracy=0.0,
        stale_memory_usage_rate=0.0,
        recall_by_type={t: 0.0 for t in EVAL_TYPES},
        precision_by_type={t: 0.0 for t in EVAL_TYPES},
        storage_rate_by_type={t: 0.0 for t in EVAL_TYPES},
        conflict_rate_by_type={t: 0.0 for t in EVAL_TYPES},
        retrieval_top1_over_time=[],
        conflict_accumulation_rate=0.0,
        goal_precision_strong=0.0,
        goal_recall_strong=0.0,
        weak_goal_rate=0.0,
        false_goal_from_wish_rate=0.0,
        false_goal_from_opinion_rate=0.0,
        false_goal_from_temporary_desire_rate=0.0,
        goal_intrusion_rate_in_retrieval_top1=0.0,
        weak_memory_usage_rate=0.0,
        weak_memory_promotion_rate=0.0,
        false_positive_reintroduced_rate=0.0,
        goal_recall_recovery_rate=0.0,
        fallback_trigger_rate=0.0,
        fallback_help_rate=0.0,
        fallback_no_effect_rate=0.0,
        fallback_harm_rate=0.0,
        avg_rank_change_from_fallback=0.0,
        weak_candidate_present_rate=0.0,
        weak_candidate_selected_rate=0.0,
        weak_coverage_rate=0.0,
        weak_coverage_goal_recall=0.0,
        weak_coverage_preference_recall=0.0,
        weak_coverage_fact_recall=0.0,
        weak_override_trigger_rate=0.0,
        weak_override_candidate_rate=0.0,
        weak_override_blocked_by_low_score_count=0,
        weak_override_blocked_by_similarity_count=0,
        weak_override_blocked_by_type_alignment_count=0,
        weak_override_success_count=0,
    )


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _best_hit_rank(ranked: list[RetrievedMemory], keywords: list[str], top_k: int) -> int:
    if not ranked:
        return max(1, top_k) + 1
    for rank, item in enumerate(ranked[: max(1, top_k)], start=1):
        if _contains_any_keyword(item.memory.content, keywords):
            return rank
    return max(1, top_k) + 1


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _group_consistency(group_to_values: dict[str, list[str]]) -> float:
    if not group_to_values:
        return 1.0

    ratios: list[float] = []
    for values in group_to_values.values():
        if not values:
            continue
        counter = Counter(values)
        majority = max(counter.values())
        ratios.append(majority / len(values))

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def _memory_type_from_label(label: str) -> MemoryType | None:
    normalized = label.lower().strip()
    if normalized in {"goal", "preference", "fact", "commitment"}:
        return MemoryType.SEMANTIC
    if normalized == "event":
        return MemoryType.EPISODIC
    if normalized == "question":
        return MemoryType.WORKING
    return None


def _memory_eval_type(memory: MemoryItem) -> str | None:
    if _is_weak_memory(memory):
        return None
    tags = {tag.lower() for tag in memory.tags}
    if "goal" in tags or memory.content.startswith("User goal:"):
        return "goal"
    if "fact" in tags or memory.content.startswith("User fact:"):
        return "fact"
    if "preference" in tags or memory.content.startswith("User preference:"):
        return "preference"
    if "commitment" in tags or memory.content.startswith("User commitment:"):
        return "commitment"
    if "event" in tags or memory.content.startswith("User event:"):
        return "event"
    return None


def _is_weak_memory(memory: MemoryItem) -> bool:
    tier = str(memory.metadata.get("memory_tier", "strong")).lower().strip()
    return tier == "weak"


def _weak_matches_intent_for_eval(memory: MemoryItem, intent: str) -> bool:
    tags = {tag.lower() for tag in memory.tags}
    if intent == "goal_recall":
        return "goal" in tags
    if intent == "preference_recall":
        return "preference" in tags
    if intent == "fact_recall":
        return "fact" in tags
    return False


def _is_goal_like_query(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["goal", "목표", "장기", "long-term"])


def _stored_expected_memory(
    retriever,
    memories: list[MemoryItem],
    source_text: str,
    expected_store_type: str,
) -> bool:
    memory_type = _memory_type_from_label(expected_store_type)
    if memory_type is None:
        return False
    label = expected_store_type.lower().strip()

    for memory in memories:
        if _is_weak_memory(memory):
            continue
        if memory.memory_type is not memory_type:
            continue
        if label in {"goal", "preference", "fact", "commitment"} and label not in {
            tag.lower() for tag in memory.tags
        }:
            continue
        similarity = retriever.text_similarity(memory.content, source_text)
        if similarity >= 0.42:
            return True
    return False


def _growth_stability(growth_history: list[int]) -> float:
    if len(growth_history) <= 1:
        return 1.0
    stdev = statistics.pstdev(growth_history)
    return 1.0 / (1.0 + stdev)


def _retrieval_degradation(
    long_top1_by_session: dict[str, list[tuple[int, bool]]],
    session_max_turn: dict[str, int],
) -> float:
    early_hits = 0
    early_total = 0
    late_hits = 0
    late_total = 0
    for session_id, points in long_top1_by_session.items():
        if not points:
            continue
        max_turn = max(session_max_turn.get(session_id, 0), 1)
        midpoint = max_turn / 2
        for turn, hit in points:
            if turn <= midpoint:
                early_total += 1
                if hit:
                    early_hits += 1
            else:
                late_total += 1
                if hit:
                    late_hits += 1

    early_rate = _safe_div(early_hits, early_total)
    late_rate = _safe_div(late_hits, late_total)
    return max(0.0, early_rate - late_rate)


def _retrieval_top1_over_time(long_top1_by_turn: dict[int, list[int]]) -> list[float]:
    if not long_top1_by_turn:
        return []
    series: list[float] = []
    for turn in sorted(long_top1_by_turn):
        values = long_top1_by_turn[turn]
        if not values:
            continue
        series.append(sum(values) / len(values))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lorenzo memory pipeline vs baseline")
    parser.add_argument("--config", type=str, default="config.example.toml")
    parser.add_argument("--scenarios", type=str, default="sample_data/eval_scenarios.json")
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios = load_scenarios(args.scenarios)

    memory_metrics = evaluate_memory_pipeline(config, scenarios)
    baseline_metrics = evaluate_baseline(scenarios)

    print(f"scenario_count={len(scenarios)}")
    print("[Memory Pipeline]")
    print(f"retrieval_hit_rate_top1={memory_metrics.retrieval_hit_rate_top1:.3f}")
    print(
        "retrieval_hit_rate_top1_strong_only="
        f"{memory_metrics.retrieval_hit_rate_top1_strong_only:.3f}"
    )
    print(
        "retrieval_hit_rate_top1_with_fallback="
        f"{memory_metrics.retrieval_hit_rate_top1_with_fallback:.3f}"
    )
    print(f"retrieval_hit_rate_top3={memory_metrics.retrieval_hit_rate_top3:.3f}")
    print(f"response_consistency={memory_metrics.response_consistency:.3f}")
    print(f"memory_precision={memory_metrics.memory_precision:.3f}")
    print(f"memory_recall={memory_metrics.memory_recall:.3f}")
    print(f"memory_recall_goal={memory_metrics.memory_recall_goal:.3f}")
    print(f"memory_recall_preference={memory_metrics.memory_recall_preference:.3f}")
    print(f"memory_recall_fact={memory_metrics.memory_recall_fact:.3f}")
    print(f"memory_growth_per_turn={memory_metrics.memory_growth_per_turn:.3f}")
    print(f"memory_growth_stability={memory_metrics.memory_growth_stability:.3f}")
    print(f"retrieval_degradation_over_time={memory_metrics.retrieval_degradation_over_time:.3f}")
    print(f"merge_activation_rate={memory_metrics.merge_activation_rate:.3f}")
    print(f"false_merge_rate={memory_metrics.false_merge_rate:.3f}")
    print(f"merge_false_reject_rate={memory_metrics.merge_false_reject_rate:.3f}")
    print(f"merge_candidate_similarity_avg={memory_metrics.merge_candidate_similarity_avg:.3f}")
    print(f"merge_rejected_reason={json.dumps(memory_metrics.merge_rejected_reason, ensure_ascii=False)}")
    print(f"rejected_count_by_type={json.dumps(memory_metrics.rejected_count_by_type, ensure_ascii=False)}")
    print(f"merge_success_count={memory_metrics.merge_success_count}")
    print(f"merge_rejected_count={memory_metrics.merge_rejected_count}")
    print(f"conflict_resolution_count={memory_metrics.conflict_resolution_count}")
    print(f"conflict_resolution_accuracy={memory_metrics.conflict_resolution_accuracy:.3f}")
    print(f"stale_memory_usage_rate={memory_metrics.stale_memory_usage_rate:.3f}")
    print(f"conflict_accumulation_rate={memory_metrics.conflict_accumulation_rate:.3f}")
    print(f"recall_by_type={json.dumps(memory_metrics.recall_by_type, ensure_ascii=False)}")
    print(f"precision_by_type={json.dumps(memory_metrics.precision_by_type, ensure_ascii=False)}")
    print(f"storage_rate_by_type={json.dumps(memory_metrics.storage_rate_by_type, ensure_ascii=False)}")
    print(f"conflict_rate_by_type={json.dumps(memory_metrics.conflict_rate_by_type, ensure_ascii=False)}")
    print(f"retrieval_top1_over_time={json.dumps(memory_metrics.retrieval_top1_over_time, ensure_ascii=False)}")
    print(f"goal_precision_strong={memory_metrics.goal_precision_strong:.3f}")
    print(f"goal_recall_strong={memory_metrics.goal_recall_strong:.3f}")
    print(f"weak_goal_rate={memory_metrics.weak_goal_rate:.3f}")
    print(f"false_goal_from_wish_rate={memory_metrics.false_goal_from_wish_rate:.3f}")
    print(f"false_goal_from_opinion_rate={memory_metrics.false_goal_from_opinion_rate:.3f}")
    print(
        "false_goal_from_temporary_desire_rate="
        f"{memory_metrics.false_goal_from_temporary_desire_rate:.3f}"
    )
    print(
        "goal_intrusion_rate_in_retrieval_top1="
        f"{memory_metrics.goal_intrusion_rate_in_retrieval_top1:.3f}"
    )
    print(f"weak_memory_usage_rate={memory_metrics.weak_memory_usage_rate:.3f}")
    print(f"weak_memory_promotion_rate={memory_metrics.weak_memory_promotion_rate:.3f}")
    print(f"false_positive_reintroduced_rate={memory_metrics.false_positive_reintroduced_rate:.3f}")
    print(f"goal_recall_recovery_rate={memory_metrics.goal_recall_recovery_rate:.3f}")
    print(f"fallback_trigger_rate={memory_metrics.fallback_trigger_rate:.3f}")
    print(f"fallback_help_rate={memory_metrics.fallback_help_rate:.3f}")
    print(f"fallback_no_effect_rate={memory_metrics.fallback_no_effect_rate:.3f}")
    print(f"fallback_harm_rate={memory_metrics.fallback_harm_rate:.3f}")
    print(f"avg_rank_change_from_fallback={memory_metrics.avg_rank_change_from_fallback:.3f}")
    print(f"weak_candidate_present_rate={memory_metrics.weak_candidate_present_rate:.3f}")
    print(f"weak_candidate_selected_rate={memory_metrics.weak_candidate_selected_rate:.3f}")
    print(f"weak_coverage_rate={memory_metrics.weak_coverage_rate:.3f}")
    print(f"weak_coverage_goal_recall={memory_metrics.weak_coverage_goal_recall:.3f}")
    print(f"weak_coverage_preference_recall={memory_metrics.weak_coverage_preference_recall:.3f}")
    print(f"weak_coverage_fact_recall={memory_metrics.weak_coverage_fact_recall:.3f}")
    print(f"weak_override_trigger_rate={memory_metrics.weak_override_trigger_rate:.3f}")
    print(f"weak_override_candidate_rate={memory_metrics.weak_override_candidate_rate:.3f}")
    print(
        "weak_override_blocked_by_low_score_count="
        f"{memory_metrics.weak_override_blocked_by_low_score_count}"
    )
    print(
        "weak_override_blocked_by_similarity_count="
        f"{memory_metrics.weak_override_blocked_by_similarity_count}"
    )
    print(
        "weak_override_blocked_by_type_alignment_count="
        f"{memory_metrics.weak_override_blocked_by_type_alignment_count}"
    )
    print(f"weak_override_success_count={memory_metrics.weak_override_success_count}")

    print("\n[Baseline]")
    print(f"retrieval_hit_rate_top1={baseline_metrics.retrieval_hit_rate_top1:.3f}")
    print(
        "retrieval_hit_rate_top1_strong_only="
        f"{baseline_metrics.retrieval_hit_rate_top1_strong_only:.3f}"
    )
    print(
        "retrieval_hit_rate_top1_with_fallback="
        f"{baseline_metrics.retrieval_hit_rate_top1_with_fallback:.3f}"
    )
    print(f"retrieval_hit_rate_top3={baseline_metrics.retrieval_hit_rate_top3:.3f}")
    print(f"response_consistency={baseline_metrics.response_consistency:.3f}")
    print(f"memory_precision={baseline_metrics.memory_precision:.3f}")
    print(f"memory_recall={baseline_metrics.memory_recall:.3f}")
    print(f"memory_recall_goal={baseline_metrics.memory_recall_goal:.3f}")
    print(f"memory_recall_preference={baseline_metrics.memory_recall_preference:.3f}")
    print(f"memory_recall_fact={baseline_metrics.memory_recall_fact:.3f}")
    print(f"memory_growth_per_turn={baseline_metrics.memory_growth_per_turn:.3f}")
    print(f"memory_growth_stability={baseline_metrics.memory_growth_stability:.3f}")
    print(f"retrieval_degradation_over_time={baseline_metrics.retrieval_degradation_over_time:.3f}")
    print(f"merge_activation_rate={baseline_metrics.merge_activation_rate:.3f}")
    print(f"false_merge_rate={baseline_metrics.false_merge_rate:.3f}")
    print(f"merge_false_reject_rate={baseline_metrics.merge_false_reject_rate:.3f}")
    print(f"merge_candidate_similarity_avg={baseline_metrics.merge_candidate_similarity_avg:.3f}")
    print(f"merge_rejected_reason={json.dumps(baseline_metrics.merge_rejected_reason, ensure_ascii=False)}")
    print(f"rejected_count_by_type={json.dumps(baseline_metrics.rejected_count_by_type, ensure_ascii=False)}")
    print(f"merge_success_count={baseline_metrics.merge_success_count}")
    print(f"merge_rejected_count={baseline_metrics.merge_rejected_count}")
    print(f"conflict_resolution_count={baseline_metrics.conflict_resolution_count}")
    print(f"conflict_resolution_accuracy={baseline_metrics.conflict_resolution_accuracy:.3f}")
    print(f"stale_memory_usage_rate={baseline_metrics.stale_memory_usage_rate:.3f}")
    print(f"conflict_accumulation_rate={baseline_metrics.conflict_accumulation_rate:.3f}")
    print(f"recall_by_type={json.dumps(baseline_metrics.recall_by_type, ensure_ascii=False)}")
    print(f"precision_by_type={json.dumps(baseline_metrics.precision_by_type, ensure_ascii=False)}")
    print(f"storage_rate_by_type={json.dumps(baseline_metrics.storage_rate_by_type, ensure_ascii=False)}")
    print(f"conflict_rate_by_type={json.dumps(baseline_metrics.conflict_rate_by_type, ensure_ascii=False)}")
    print(f"retrieval_top1_over_time={json.dumps(baseline_metrics.retrieval_top1_over_time, ensure_ascii=False)}")
    print(f"goal_precision_strong={baseline_metrics.goal_precision_strong:.3f}")
    print(f"goal_recall_strong={baseline_metrics.goal_recall_strong:.3f}")
    print(f"weak_goal_rate={baseline_metrics.weak_goal_rate:.3f}")
    print(f"false_goal_from_wish_rate={baseline_metrics.false_goal_from_wish_rate:.3f}")
    print(f"false_goal_from_opinion_rate={baseline_metrics.false_goal_from_opinion_rate:.3f}")
    print(
        "false_goal_from_temporary_desire_rate="
        f"{baseline_metrics.false_goal_from_temporary_desire_rate:.3f}"
    )
    print(
        "goal_intrusion_rate_in_retrieval_top1="
        f"{baseline_metrics.goal_intrusion_rate_in_retrieval_top1:.3f}"
    )
    print(f"weak_memory_usage_rate={baseline_metrics.weak_memory_usage_rate:.3f}")
    print(f"weak_memory_promotion_rate={baseline_metrics.weak_memory_promotion_rate:.3f}")
    print(f"false_positive_reintroduced_rate={baseline_metrics.false_positive_reintroduced_rate:.3f}")
    print(f"goal_recall_recovery_rate={baseline_metrics.goal_recall_recovery_rate:.3f}")
    print(f"fallback_trigger_rate={baseline_metrics.fallback_trigger_rate:.3f}")
    print(f"fallback_help_rate={baseline_metrics.fallback_help_rate:.3f}")
    print(f"fallback_no_effect_rate={baseline_metrics.fallback_no_effect_rate:.3f}")
    print(f"fallback_harm_rate={baseline_metrics.fallback_harm_rate:.3f}")
    print(f"avg_rank_change_from_fallback={baseline_metrics.avg_rank_change_from_fallback:.3f}")
    print(f"weak_candidate_present_rate={baseline_metrics.weak_candidate_present_rate:.3f}")
    print(f"weak_candidate_selected_rate={baseline_metrics.weak_candidate_selected_rate:.3f}")
    print(f"weak_coverage_rate={baseline_metrics.weak_coverage_rate:.3f}")
    print(f"weak_coverage_goal_recall={baseline_metrics.weak_coverage_goal_recall:.3f}")
    print(
        "weak_coverage_preference_recall="
        f"{baseline_metrics.weak_coverage_preference_recall:.3f}"
    )
    print(f"weak_coverage_fact_recall={baseline_metrics.weak_coverage_fact_recall:.3f}")
    print(f"weak_override_trigger_rate={baseline_metrics.weak_override_trigger_rate:.3f}")
    print(f"weak_override_candidate_rate={baseline_metrics.weak_override_candidate_rate:.3f}")
    print(
        "weak_override_blocked_by_low_score_count="
        f"{baseline_metrics.weak_override_blocked_by_low_score_count}"
    )
    print(
        "weak_override_blocked_by_similarity_count="
        f"{baseline_metrics.weak_override_blocked_by_similarity_count}"
    )
    print(
        "weak_override_blocked_by_type_alignment_count="
        f"{baseline_metrics.weak_override_blocked_by_type_alignment_count}"
    )
    print(f"weak_override_success_count={baseline_metrics.weak_override_success_count}")


if __name__ == "__main__":
    main()
