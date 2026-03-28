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
from lorenzo.models import MemoryItem, MemoryType
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

            top1_text = (result.retrieved_memories[0].memory.content if result.retrieved_memories else "").lower()
            top3_text = " ".join(item.memory.content for item in result.retrieved_memories[:3]).lower()
            top1_type = (
                _memory_eval_type(result.retrieved_memories[0].memory)
                if result.retrieved_memories
                else None
            )

            if scenario.expected_retrieval_keywords:
                retrieval_total += 1
                top1_hit = _contains_any_keyword(top1_text, scenario.expected_retrieval_keywords)
                top3_hit = _contains_any_keyword(top3_text, scenario.expected_retrieval_keywords)
                if top1_hit:
                    retrieval_hits_top1 += 1
                if top3_hit:
                    retrieval_hits_top3 += 1

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

    return EvalMetrics(
        retrieval_hit_rate_top1=retrieval_hit_rate_top1,
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
    )


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


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

    print("\n[Baseline]")
    print(f"retrieval_hit_rate_top1={baseline_metrics.retrieval_hit_rate_top1:.3f}")
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


if __name__ == "__main__":
    main()
