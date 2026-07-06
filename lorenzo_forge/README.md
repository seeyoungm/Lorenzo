# Lorenzo Forge

데이터 프로파일을 입력하면, **학습된 메타 신경망(meta-model)**이 적합한 신경망 아키텍처를 추천해주는 프로토타입입니다.
"AI가 신경망 설계를 생성한다"는 개념을 아주 작은 스케일의 NAS(Neural Architecture Search) 방식으로 구현했습니다.

## 핵심 아이디어

Lorenzo Forge는 규칙 기반 추천기가 아닙니다. 실제로 학습되는 신경망(메타 모델)이 "어떤 데이터에 어떤 아키텍처가 잘 맞는지"를
경험적 탐색 결과로부터 학습합니다.

```
1) 무작위 데이터 프로파일 생성 (tabular / image, 크기, 클래스 수, 노이즈 등)
2) 각 프로파일에 대해 실제 후보 아키텍처 여러 개를 빠르게 학습·평가 (random search)
3) 가장 성능이 좋았던 아키텍처를 "정답 라벨"로 기록
   -> (프로파일 특징, 최적 아키텍처) 쌍으로 구성된 학습 코퍼스 완성
4) 이 코퍼스로 메타 신경망을 지도학습 (multi-head classification)
5) 새 데이터 프로파일이 들어오면, 메타 신경망이 탐색 없이 즉시 아키텍처를 추천
```

즉 2~3단계(탐색)는 학습 데이터를 만들기 위한 비용이고, 진짜 결과물은 4단계에서 학습된 메타 신경망입니다.
탐색 없이도 "감"으로 바로 추천할 수 있다는 게 핵심입니다.

## 출력 형태

추천 결과는 **바로 실행 가능한 코드가 아니라 구조 명세(JSON)** 로 출력됩니다.

```json
{
  "architecture": {
    "task_type": "tabular",
    "blocks": [{"type": "dense", "units": 128, "activation": "tanh"}],
    "dropout": 0.2,
    "optimizer": "adam",
    "learning_rate": 0.01
  },
  "confidence": {"num_blocks": 0.40, "units": 0.32, "...": "..."}
}
```

`confidence`는 메타 모델의 각 결정(레이어 수, 유닛 수, 활성화 함수 등)에 대한 softmax 확신도입니다.

## 지원 범위 (v0.1)

- task type: `tabular` (MLP), `image` (CNN)
- 탐색 축: 블록 수, 유닛/필터 수, (이미지) 커널 크기, 활성화 함수, dropout, optimizer, learning rate
- 데이터: 실제 데이터셋 또는 합성 데이터 모두 프로파일화 가능 (`DataProfile.from_arrays`)

## 빠른 시작

```bash
cd lorenzo_forge
pip install -e ".[dev]"

# 1) 탐색 코퍼스 생성 (프로파일마다 후보 아키텍처를 실제로 학습해서 라벨 생성)
lorenzo-forge build-corpus --profiles 80 --candidates 8 --search-epochs 4

# 2) 메타 모델 학습
lorenzo-forge train-meta --epochs 30

# 3) 새 데이터 프로파일에 대한 아키텍처 추천
lorenzo-forge recommend --task tabular --input-dim 30 --output-dim 4 --num-samples 5000
```

## 실험 기록 — v0.1 (음성 결과)

첫 풀 규모 실행 결과, **현재 설정으로는 메타 모델이 학습에 실패**했습니다. 정직하게 기록합니다.

**실행 조건**
- `build-corpus --profiles 80 --candidates 8 --search-epochs 4` (코퍼스 생성 ~46분)
- `train-meta --epochs 30`, held-out 20% 평가

**held-out per-head 정확도 (메타 모델 vs 다수결 baseline)**

| head | meta | baseline | 판정 |
|---|---|---|---|
| num_blocks | 0.20 | 0.35 | ❌ |
| units | 0.30 | 0.30 | ➖ |
| kernel | 0.75 | 0.80 | ❌ |
| activation | 0.30 | 0.70 | ❌ |
| dropout | 0.25 | 0.45 | ❌ |
| optimizer | 0.85 | 0.85 | ➖ |
| lr | 0.50 | 0.60 | ❌ |

모든 헤드에서 "가장 흔한 값을 그냥 찍는" 다수결 baseline과 같거나 낮음 → 유의미한 학습 신호를 얻지 못함.

**원인 진단**
1. **이미지 라벨이 사실상 노이즈** (가장 결정적): 이미지 프로파일 38개 **전부**가 후보 아키텍처 최고 점수 ≥0.99에 도달. 합성 이미지 태스크가 너무 쉬워 어떤 아키텍처든 100% 근처로 풀리고, 기록된 "최적 아키텍처"는 동률 중 무작위 선택 → 데이터 절반이 랜덤 라벨.
2. **데이터 부족**: 80개 → train 60 / test 20. 7개 헤드 매핑 학습에 부족.
3. **치우친 라벨 분포로 baseline이 강함**: optimizer 65/80 adam, activation 46/80 relu 등. 다수결만으로 0.7~0.85 → 진짜 신호 없이는 못 이김.

**개선 계획** (미적용)
- 라벨 품질: 합성 이미지 태스크 난이도 상향(노이즈↑/신호↓) + 동률 시 "더 작고 빠른 모델 우선" tie-break로 라벨을 결정적·의미 있게.
- 메타 모델 설계: "우승자 1개 분류" → "후보 N개 점수 회귀/랭킹"으로 전환(프로파일당 신호 N배, 동률 문제 자연 해소).
- 규모: 프로파일 수 대폭 확대(수백~).

> 파이프라인(탐색 → 라벨 생성 → 메타 학습 → 저장)은 end-to-end로 정상 동작함이 확인됨. 실패한 것은 결과물의 품질이며, 원인은 위와 같이 규명됨. 현재 `artifacts/`의 메타 모델은 실사용 가치가 없어 저장소에 커밋하지 않음(gitignore).

## 프로젝트 구조

```
lorenzo_forge/
  profile.py            # DataProfile: 데이터 특징 벡터 정의
  search_space.py        # ArchitectureSpec: 탐색 가능한 아키텍처 인코딩/디코딩
  synthetic.py            # 합성 프로파일/데이터셋 생성 (학습 라벨 확보용)
  candidate_trainer.py    # 후보 아키텍처를 실제로 빌드/학습/평가
  search.py               # 프로파일별 random search로 최적 아키텍처 탐색
  dataset_builder.py       # 탐색을 반복해 메타 모델용 학습 코퍼스 생성
  meta_model.py            # 실제 학습되는 멀티헤드 메타 신경망 + 추천 함수
  cli.py                   # lorenzo-forge CLI
tests_forge/               # 빠른 파라미터로 전체 파이프라인 검증
```

## 비목표 (v0.1)

- 실행 가능한 학습 코드 자동 생성 (구조 명세까지만 출력)
- weight-sharing supernet, 진화 탐색 등 고급 NAS 기법
- 시퀀스/텍스트(RNN/Transformer) 태스크 지원
