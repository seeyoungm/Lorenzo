# Lorenzo v1

Memory-Centric + Modular Intelligence 구조를 가진 Python AI 프로토타입입니다.

## v1.2 Status

Key improvements:
- strict goal classification (`strong_goal` vs `weak_goal`)
- commitment precision improved to `1.0`
- elimination of false goal memories (goal boundary eval 기준)
- expanded evaluation (`380` scenarios, `270`-turn long sessions)

Trade-offs:
- stricter storage policy로 인해 `retrieval_top1` 하락

Status:
- memory semantics stabilized
- next focus: retrieval recall recovery

## v1.3 Update

v1.3 beta:
- Added conditional weak override for fallback retrieval
- Diagnosed weak-override blocking causes
- Improved broad-eval top1 retrieval from 0.562 to 0.594
- Preserved precision and stale-memory safety
- No false-positive reintroduction observed

## v2-alpha Update

Iterative Reasoning Engine (2-pass) 추가:
- pass 1: 초기 retrieval + draft answer
- pass 2: `original query + draft answer` 기반 re-retrieval + refinement
- refinement trigger:
  - low confidence
  - conflicting memory
  - insufficient supporting memory
  - answer-memory mismatch
- max iteration cap (`2~3`) 적용

v2-alpha broad eval snapshot:
- `retrieval_hit_rate_top1_with_fallback=0.898`
- `fallback_harm_rate=0.000`
- `false_positive_reintroduced_rate=0.000`
- `stale_memory_usage_rate=0.000`
- `conflict_resolution_accuracy=1.000`
- `refinement_improvement_rate=0.746`
- `answer_change_rate=0.746`
- `iteration_gain_score=0.055`

## 핵심 특징

- 모듈형 파이프라인
  - `Input Processor`
  - `Memory Module`
  - `Reasoning Module`
  - `Language Module`
  - `Orchestrator`
- Persistent memory (`JSONL` 기반)
- 응답 생성 전 메모리 검색 필수 수행
- 검색 점수: `embedding semantic + recency + importance + lexical fallback`
- 검색 보강: `score normalization + intent/type alignment + access boost + conflict penalty + diversity penalty + retrieval_reason + embedding cache`
- iterative reasoning:
  - objective-routed refinement (`fact gap / preference mismatch / conflict / support completion`)
  - error-focused re-query (problematic claim + unresolved memory type 기반)
- 다국어 유사도 지원(한국어/영어 개념 정규화 기반)
- memory type 구분: `episodic / semantic / working`
- 저장 정책: `importance threshold(상향) + deduplication + pre-write semantic summary synthesis + type filter`
- two-tier memory 정책:
  - `strong memory`: fully trusted memory
  - `weak memory`: retrieval fallback 힌트 전용 memory (`weak_goal`, `weak_preference`, `weak_fact`)
- intent 경계 규칙 강화:
  - `goal`: `strong_goal / weak_goal` 구분(semantic 저장은 strong만)
    - strong 조건: 미래 지향 + 달성/변환 의도 + 비일시적 지속성
    - strong 제외: wish / opinion / temporary desire / conversational meta
  - `preference`: 스타일/행동 선호 (commitment와 분리)
  - `commitment`: 명시적 약속/미래 실행 선언/스케줄 액션
- conflict resolution 정책(타입별):
  - `goal`: latest wins + history 보존
  - `preference`: latest wins + 이전값 inactive history
  - `fact`: 자동 overwrite 금지, keep-both + conflict 마킹
  - `commitment`: 명시적 confirmation 신호 없으면 overwrite 금지
  - `event`: timestamp 기준 분리 보존
- 교체 가능한 language backend (`rule_based`, `echo`)
- CLI 데모 제공
- baseline 비교 평가 CLI (`lorenzo-eval`)
- 단위 테스트 포함

## 프로젝트 구조

```
lorenzo/
  __init__.py
  cli.py
  config.py
  eval.py
  interfaces.py
  input_processor.py
  models.py
  orchestrator.py
  memory/
    store.py
    retriever.py
  reasoning/
    planner.py
    iterative.py
  language/
    adapter.py
    backends.py
tests/
sample_data/
  eval_scenarios.json
  eval_scenarios_broad.json
  eval_scenarios_goal_refinement.json
config.example.toml
```

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1) 단발 실행

```bash
lorenzo --config config.example.toml --once "나는 장기 기억이 있는 AI를 만들고 싶어"
```

### 2) 대화형 CLI

```bash
lorenzo --config config.example.toml
```

- 종료: `/exit`
- 현재 memory item 수: `/count`

## 동작 흐름

1. Input Processor가 입력을 분류
2. Memory Module이 저장된 기억을 검색 (항상 선행)
3. Reasoning Planner가 전략 선택
4. Language Adapter가 응답 생성
5. Memory update rule로 episodic/semantic/working memory 갱신

## 설정

`config.example.toml`

```toml
[memory]
path = "sample_data/memory_store.jsonl"
top_k = 5
retrieval_preselect_multiplier = 3
min_similarity_floor = 0.10
importance_weight = 0.10
recency_weight = 0.15
similarity_weight = 0.75
lexical_fallback_weight = 0.05
recency_half_life_hours = 72.0
min_importance_to_store = 6.5
dedup_similarity_threshold = 0.97
semantic_merge_similarity_threshold = 0.60
merge_confidence_threshold = 0.68

[language]
backend = "rule_based"
```

## 테스트

```bash
pytest -q
```

## 평가 (Baseline 비교)

```bash
lorenzo-eval --config config.example.toml --scenarios sample_data/eval_scenarios.json
```

평가셋 구성:
- 기본: `sample_data/eval_scenarios.json`
- 확장: `sample_data/eval_scenarios_broad.json` (536 시나리오, 100+ turn long session 포함)
- goal 경계 전용: `sample_data/eval_scenarios_goal_refinement.json` (24 시나리오)
- paraphrase / cross-language / misleading / conflicting memory / merge-stress / boundary 케이스 포함

출력 지표:
- `retrieval_hit_rate_top1`
- `retrieval_hit_rate_top3`
- `response_consistency`
- `memory_precision`
- `memory_recall` (overall + by type: goal/preference/fact)
- `memory_growth_per_turn`
- `memory_growth_stability`
- `retrieval_degradation_over_time`
- `merge_activation_rate`
- `false_merge_rate`
- `merge_false_reject_rate`
- `merge_candidate_similarity_avg`
- `merge_rejected_reason`
- `rejected_count_by_type`
- `merge_success_count`
- `merge_rejected_count`
- `conflict_resolution_count`
- `conflict_resolution_accuracy`
- `stale_memory_usage_rate`
- `recall_by_type` (`goal/fact/preference/commitment/event`)
- `precision_by_type`
- `storage_rate_by_type`
- `conflict_rate_by_type`
- `retrieval_top1_over_time`
- `conflict_accumulation_rate`
- `goal_precision_strong`
- `goal_recall_strong`
- `weak_goal_rate`
- `false_goal_from_wish_rate`
- `false_goal_from_opinion_rate`
- `false_goal_from_temporary_desire_rate`
- `goal_intrusion_rate_in_retrieval_top1`
- `retrieval_hit_rate_top1_strong_only`
- `retrieval_hit_rate_top1_with_fallback`
- `weak_memory_usage_rate`
- `weak_memory_promotion_rate`
- `false_positive_reintroduced_rate`
- `goal_recall_recovery_rate`
- `refinement_improvement_rate`
- `conflict_detected_rate`
- `answer_change_rate`
- `iteration_gain_score`
- `factual_refinement_gain`
- `preference_alignment_gain`
- `support_completion_gain`
- `conflict_fix_rate`

## 구현 우선순위 매핑

- Phase 1
  - 데이터 모델 정의: `lorenzo/models.py`
  - memory store/retrieval: `lorenzo/memory/*`
  - orchestrator 기본 흐름: `lorenzo/orchestrator.py`
- Phase 2
  - reasoning planner: `lorenzo/reasoning/planner.py`
  - language adapter: `lorenzo/language/*`
  - memory update 규칙: `lorenzo/orchestrator.py::_update_memories`
- Phase 3
  - CLI 데모: `lorenzo/cli.py`
  - 테스트: `tests/*`
  - 샘플 세션 데이터: `sample_data/memory_store.jsonl`

## 비목표

- 자체 LLM 학습
- 멀티모달
- 강화학습
- 대규모 분산/GPU 최적화
