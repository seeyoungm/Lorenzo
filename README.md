# Lorenzo v1

Memory-Centric + Modular Intelligence 구조를 가진 Python AI 프로토타입입니다.

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
- 다국어 유사도 지원(한국어/영어 개념 정규화 기반)
- memory type 구분: `episodic / semantic / working`
- 저장 정책: `importance threshold(상향) + deduplication + pre-write semantic summary synthesis + type filter`
- conflict resolution 정책: `latest_wins + conflict_history` (fact 충돌 시)
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
  language/
    adapter.py
    backends.py
tests/
sample_data/
  eval_scenarios.json
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
- 총 168 시나리오
- long session 120턴 포함 (noise/contradiction/paraphrase drift)
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
- `merge_success_count`
- `merge_rejected_count`
- `conflict_resolution_count`
- `conflict_resolution_accuracy`
- `stale_memory_usage_rate`

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
