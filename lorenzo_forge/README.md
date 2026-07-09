# Lorenzo Forge

데이터 프로파일을 입력하면, **학습된 메타 신경망(meta-model)**이 적합한 신경망 아키텍처를 추천해주는 프로토타입입니다.
"AI가 신경망 설계를 생성한다"는 개념을 아주 작은 스케일의 NAS(Neural Architecture Search) 방식으로 구현했습니다.

## 핵심 아이디어

Lorenzo Forge는 규칙 기반 추천기가 아닙니다. 실제로 학습되는 신경망(메타 모델)이 "어떤 데이터에 어떤 아키텍처가 잘 맞는지"를
경험적 탐색 결과로부터 학습합니다.

```
1) 무작위 데이터 프로파일 생성 (tabular / image, 크기, 클래스 수, 노이즈 등)
2) 각 프로파일에 대해 실제 후보 아키텍처 여러 개를 빠르게 학습·평가 (random search)
3) 후보별 (아키텍처, 실측 정확도) 쌍을 전부 기록
   -> (프로파일 특징 + 아키텍처 인코딩, 정확도) 학습 코퍼스 완성
4) 이 코퍼스로 스코어러 메타 신경망을 회귀 학습
   : (프로파일 + 아키텍처) -> 예상 정확도
5) 새 프로파일이 들어오면, 검색공간 전체를 열거·점수화해 최고점 아키텍처를 추천 (탐색 없이)
```

즉 2~3단계(탐색)는 학습 데이터를 만들기 위한 비용이고, 진짜 결과물은 4단계에서 학습된 스코어러입니다.
탐색 없이도 "감"으로 바로 아키텍처를 랭킹·추천할 수 있다는 게 핵심입니다.

> **설계 노트 (v0.1 → v0.2)**: 초기 v0.1은 프로파일당 "우승자 1개"를 분류하는 방식이었는데,
> 라벨이 프로파일당 1개뿐이라 데이터가 부족했고 동률일 때 라벨이 무작위가 됐습니다(아래 실험 기록 참고).
> v0.2에서 **후보 전부의 점수를 회귀/랭킹**하는 스코어러로 전환해 프로파일당 라벨이 N배가 되고
> 동률 문제도 자연히 해소됐습니다.

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
  "predicted_accuracy": 0.91,
  "top_k": [{"arch": "...", "predicted_accuracy": 0.91}, "..."]
}
```

`predicted_accuracy`는 스코어러가 예측한 해당 아키텍처의 기대 정확도이고, `top_k`는 상위 후보 랭킹입니다.

## 지원 범위

- task type: `tabular`(MLP), `image`(CNN), `text`(Embedding+RNN/Conv1D), `timeseries`(RNN/Conv1D)
- 탐색 축: 블록 수, 유닛/필터 수, 커널 크기, (텍스트)임베딩 차원, (시퀀스)인코더 종류(lstm/gru/conv1d), 활성화, dropout, optimizer, lr
- 코퍼스 데이터: tabular·timeseries는 합성, image·text는 실제 데이터셋(MNIST·Fashion / IMDB·Reuters)
- 임의 데이터셋도 프로파일화 가능 (`DataProfile.from_arrays`)

## 빠른 시작

```bash
cd lorenzo_forge
pip install -e ".[dev]"

# 1) 탐색 코퍼스 생성 (프로파일마다 후보 아키텍처를 실제로 학습해서 라벨 생성; tabular/image/text)
lorenzo-forge build-corpus --profiles 90 --candidates 6 --search-epochs 4

# 2) 메타 모델(스코어러) 학습
lorenzo-forge train-meta --epochs 30

# 3) 새 데이터 프로파일에 대한 아키텍처 추천
lorenzo-forge recommend --task tabular --input-dim 30 --output-dim 4 --num-samples 5000

# 4) 릴리스: 추천 top-k를 실제로 풀 학습해 최고를 골라 패키징 (Phase 1)
lorenzo-forge release --name "Lorenzo Image 1.0" --domain mnist \
    --scorer lorenzo_forge/artifacts/scorer_model.keras --top-k 5 --epochs 40 \
    --out-dir lorenzo_forge/releases/image_1_0
# 텍스트 릴리스: --domain imdb (또는 reuters)
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

**개선 계획** (→ v0.2에서 적용, 아래 참조)
- 라벨 품질: 합성 이미지 태스크 난이도 조정 + 동률 시 "더 작고 빠른 모델 우선" tie-break로 라벨을 결정적·의미 있게.
- 메타 모델 설계: "우승자 1개 분류" → "후보 N개 점수 회귀/랭킹"으로 전환(프로파일당 신호 N배, 동률 문제 자연 해소).

## 실험 기록 — v0.2 (개선 적용 후)

v0.1이 진단한 두 원인을 모두 수정했습니다.
- **라벨 품질(A)**: 합성 이미지를 "약하고 국소적인 클래스 패치 + 공유 노이즈 배경"으로 재설계(어떤 아키텍처든 1.0 포화되지 않도록) + 동률 후보 중 가장 단순한 것을 라벨로 tie-break.
- **메타 모델(B)**: 단일 우승자 분류기 → **스코어러 회귀**(프로파일+아키텍처 → 예상 정확도), 프로파일당 후보 전부로 학습. 추천은 검색공간 전체를 점수화해 최고점 선택.

**실행 조건**: `--profiles 80 --candidates 8 --search-epochs 4` (코퍼스 ~29분), 스코어러 200 epochs, 프로파일 단위 held-out 20%.

**코퍼스 건강도**: 포화(≥0.99) **0/80** — v0.1의 라벨 노이즈 버그 완전 해결. tabular 프로파일 내 점수 편차 0.504(강한 신호), image 0.114(약하지만 존재).

**held-out 랭킹 품질** (스코어러가 unseen 프로파일에서 아키텍처를 얼마나 잘 줄 세우는가)

| 구분 | Spearman(pred,실측) | top-1 regret | 랜덤 regret | top-1 적중 | 판정 |
|---|---|---|---|---|---|
| overall (n=20) | 0.549 | 0.013 | 0.178 | 0.60 | 랜덤 대비 **14×** |
| **tabular (n=9)** | **0.913** | 0.004 | 0.351 | **0.889** | 랜덤 대비 **83×** ✅ |
| image (n=11) | 0.251 | 0.020 | 0.103 | 0.364 | 랜덤 대비 5× (약함) |

- **regret** = (그 프로파일에서 실측된 최적 정확도) − (스코어러가 고른 아키텍처의 실측 정확도). 0에 가까울수록 추천이 최적에 근접.
- **tabular은 거의 완벽하게 랭킹**(Spearman 0.91). 스코어러 추천을 따르면 최적 대비 0.4%p만 손해(랜덤은 35%p), 정확한 최적을 89% 적중.
- **image는 여전히 약함**(Spearman 0.25). 합성 이미지 태스크에선 아키텍처 선택이 본질적으로 성능을 덜 가르기(편차 0.114) 때문. 그래도 랜덤보다 5배 낫고 더 이상 망가지지 않음. → v0.3에서 해결.

## 실험 기록 — v0.3 (실제 데이터셋 이미지)

v0.2의 남은 과제(이미지 신호 약함)를 해결했습니다. 원인은 **합성 이미지 태스크가 본질적으로 아키텍처를 덜 가른다**는 것이었으므로, 이미지 프로파일을 **실제 데이터셋에서** 생성하도록 바꿨습니다(`real_image.py`).

- 이미지 프로파일을 **MNIST / Fashion-MNIST**에서 샘플링. 클래스 부분집합·샘플 수·해상도(28/20/14)·추가 노이즈를 무작위로 바꿔 프로파일 다양성 확보.
- 실제 데이터에선 좋은 conv가 나쁜 것을 **0.4~0.7 정확도 차이**로 앞섬(합성은 ~0.09) → 진짜 랭킹 신호.
- 이미지 후보는 search epoch를 2배(8)로 학습(실제 이미지는 아키텍처가 갈리는 데 더 필요).
- tabular은 계속 합성(scikit-learn `make_classification`).

**코퍼스 건강도**: 이미지 프로파일 내 점수 편차 **0.114 → 0.613(5배)**. 아키텍처가 실제로 성능을 가름.

**held-out 랭킹 품질**

| 구분 | Spearman | top-1 regret | 랜덤 regret | top-1 적중 | v0.2 → v0.3 |
|---|---|---|---|---|---|
| overall (n=20) | **0.834** | 0.046 | 0.240 | 0.60 | 0.55 → 0.83 |
| tabular (n=12) | 0.877 | 0.009 | 0.171 | 0.58 | 강세 유지 |
| **image (n=8)** | **0.769** | 0.102 | 0.346 | 0.625 | **0.25 → 0.77** 🚀 |

- **이미지 Spearman 0.25 → 0.77**로 3배 개선, tabular(0.88)에 근접. 이미지에서도 스코어러가 아키텍처를 제대로 줄 세움.
- 이미지 regret(0.10)이 tabular(0.009)보다 큰 것은 실제 이미지 점수 범위가 0.13~1.00으로 넓어 최적을 조금만 빗나가도 절대 손실이 크기 때문. 랭킹 자체는 랜덤 대비 3.4배 우수.

**결론**: tabular·image **양쪽 모두 실사용 가치가 있는 아키텍처 추천기**가 됨(overall Spearman 0.83). 이미지 신호는 실제 데이터 도입으로 확보. 다음 확장 후보: 텍스트(시퀀스) 태스크 → v0.4.

## 실험 기록 — v0.4 (텍스트 도메인 추가, 3-도메인)

세 번째 도메인 **텍스트(시퀀스 분류)** 를 추가. 완전히 다른 모델 계열(Embedding → LSTM/GRU/Conv1D → Dense)이라 검색공간에 `embedding_dim`·`encoder` 축, 프로파일에 `vocab_size`·`is_text` 특징을 추가하고 스코어러를 재학습.

- 텍스트 프로파일은 실제 **IMDB(감성)/Reuters(토픽)** 에서 어휘·시퀀스길이·클래스·샘플수를 변주.
- 프로브 결과 conv1d-adam(≈0.75)이 recurrent/sgd를 확실히 앞서 랭킹 신호 존재. 단 신호는 tabular/image보다 약함(프로파일 내 편차 0.25).

**held-out 랭킹 품질** (90 프로파일, 6 후보, held-out 23)

| 도메인 | Spearman | regret | 랜덤 regret | top-1 적중 |
|---|---|---|---|---|
| overall | 0.648 | 0.027 | 0.191 | 0.57 |
| image | **0.916** | **0.000** | 0.332 | **1.0** |
| tabular | 0.735 | 0.032 | 0.109 | 0.25 |
| **text (신규)** | 0.392 | 0.042 | 0.153 | 0.556 |

- **텍스트 도메인이 추가돼 작동함** — Spearman 0.39로 셋 중 가장 약하지만, 추천을 따르면 최적 대비 4%p 손해로 **랜덤 대비 3.7배** 우수, top-1 56% 적중.
- 텍스트가 약한 이유: recurrent 학습 노이즈 + 프로파일 내 아키텍처 편차가 작음(0.25). 릴리스 단계의 실측 검증이 이를 보완(아래 Text 1.0 참고).
- image는 이 실행에서 held-out 완벽(Spearman 0.92, regret 0). tabular은 후보 축소(8→6)와 3-도메인 용량 분할로 이전보다 소폭 하락하나 regret은 여전히 작음.

## 실험 기록 — v0.5 (시계열 도메인 추가, 4-도메인)

네 번째 도메인 **시계열(time-series 분류)** 추가. 텍스트의 인코더 축(lstm/gru/conv1d)을 재사용하되 **임베딩 없이 실수 시퀀스(timesteps×channels)를 직접** 인코더에 투입. 합성 시계열은 클래스별 **전역 주파수 + 위치 무관 국소 모티프**로 구성 → 시간 모델링이 진짜 필요.

**held-out 랭킹 품질** (120 프로파일, 6 후보, held-out 30)

| 도메인 | Spearman | regret | 랜덤 regret | top-1 적중 |
|---|---|---|---|---|
| overall | 0.743 | 0.022 | 0.306 | 0.70 |
| tabular | 0.872 | 0.010 | 0.229 | **0.909** |
| image | 0.806 | 0.012 | 0.207 | 0.60 |
| **timeseries (신규)** | **0.784** | 0.033 | 0.154 | 0.625 |
| text | 0.402 | 0.040 | 0.061 | 0.50 |

- **시계열이 강하게 안착** — Spearman 0.784로 image(0.81)에 근접, 텍스트(0.40)보다 훨씬 강함. 프로파일 내 아키텍처 점수 편차 **0.664로 4개 도메인 중 최고**(합성 시계열이 진짜 신호를 줌).
- 시계열에선 conv1d/gru가 프로파일에 따라 번갈아 우세 → 스코어러가 인코더 선택을 실제로 학습.
- overall Spearman이 v0.4의 0.65 → **0.74**로 오름(새 도메인이 오히려 전체 신호를 강화). 스코어러 입력은 36차원(4개 task flag + `vocab_size` 등)으로 확장, `artifacts/scorer_model.keras`가 4-도메인 버전으로 갱신됨.

## 실험 기록 — v0.6 (bidirectional 인코더, 텍스트 근본 개선)

텍스트의 가장 약한 랭킹(Spearman 0.40)을 근본 개선. 원인은 recurrent 인코더가 **단방향뿐**이라 conv1d에 밀린 것 → 시퀀스 인코더 축에 **`bilstm`/`bigru`** 추가(이제 5종). 프로브(IMDB 8 epoch): bigru 0.792 / bilstm 0.789 > conv1d 0.777.

**held-out 랭킹 품질** (120 프로파일, 텍스트 8 epoch)

| 도메인 | v0.5 Spearman | v0.6 Spearman |
|---|---|---|
| **text** | 0.402 | **0.578** ⬆ |
| image | 0.806 | 0.892 |
| timeseries | 0.784 | 0.748 |
| tabular | 0.872 | 0.780 |
| overall | 0.743 | 0.745 |

- **텍스트 랭킹 0.40 → 0.58로 실질 개선.** 우승 인코더 분포(텍스트 26개): conv1d 9, **bigru 9 + bilstm 2 + lstm 5** — bidirectional 계열이 conv1d만큼 채택됨. 시계열도 bigru/bilstm이 우세.
- 비용: bidirectional은 학습이 ~4배 느려 코퍼스 재빌드가 약 4시간(텍스트 병목). 개선은 실질적이나 도메인별 held-out 표본이 작아(텍스트 n=6) 일부 지표는 노이지 — Spearman이 신뢰 지표.
- 스코어러 입력 38차원, 텍스트 열거 10368 / 시계열 3456 아키텍처.

**릴리스 영향(Text 1.2)**: 새 스코어러로 텍스트를 재릴리스하니 **우승이 bigru로 바뀜**(bidirectional 채택됨). 하지만 정확도는 IMDB 0.838→0.836, Reuters 0.803→0.809로 **사실상 평평**. 즉 bidirectional은 **랭킹 품질과 선택 능력은 올렸지만 정확도 천장은 못 올림**(bigru ≈ conv1d). 이 규모에서 정확도를 더 뚫으려면 사전학습 임베딩(GloVe)이나 Transformer가 필요 — 비용 대비 수익이 커 현 범위 밖. **교훈: "근본 개선"이 항상 정확도로 직결되진 않으며, 정직하게 계측해야 안다.**

## 실험 기록 — v0.7 (BatchNorm / 잔차 블록 / global-average-pooling)

검색공간에 두 축을 추가: **`block_style`**(plain vs. 2-conv/dense 잔차 블록, 채널 불일치 시 1×1/1-unit projection shortcut)와 **`pool_style`**(기존 flatten+dense / max-pool vs. global-average-pooling). BatchNorm은 축 추가 없이 tabular Dense·image Conv2D·conv1d 인코더에 무조건 적용(재귀 인코더는 대상 밖). 스코어러 입력 38→**42차원**.

코퍼스 120 프로파일 재빌드 후 재릴리스한 결과(top-k를 5~12로 넓혀 실측 검증):

| 도메인 | 이전 | 신규 | 비고 |
|---|---|---|---|
| Tabular | 0.892 | **0.922** | 채택 |
| TimeSeries | 0.867 | **0.880** | 채택 |
| Image | 0.990 | 0.983 (미채택) | 예측이 top-10 전부 0.85 안팎으로 몰려 num_blocks=1 계열만 뽑힘; residual 후보는 (256유닛, k5, 풀 MNIST) 후보당 50분+ 걸려 top-k 확장 비용이 큼 |
| Text (IMDB) | 0.867 | 0.861 (미채택) | bigru 계열 일부가 실측 시 val/test ≈ 0.5 이하로 붕괴 |
| Text (Reuters) | 0.818 | 0.794 (미채택) | 위와 동일 패턴 |

**스코어러 드리프트가 다시 확인됨**: 스코어러를 재학습할 때마다 도메인별로 이전 최적 아키텍처가 top-k 밖으로 밀려날 수 있음(Tabular 1.0→1.1 사이에도 관찰됐던 현상). 개선이 보장되지 않는다는 뜻이며, 릴리스 파이프라인이 실측으로 검증해 하락을 자동으로 걸러내는 게 핵심 안전장치임을 재확인. CIFAR-10(컬러 이미지) 도메인 추가도 함께 시도했으나 배포처(`cs.toronto.edu`)가 이 네트워크에서 초당 수백 바이트 수준이라 보류.

## Phase 1 — 릴리스 파이프라인 (`forge-release`)

스코어러는 아키텍처를 **랭킹**만 합니다. 릴리스하려면 **실제로 학습되고 패키징된 모델**이 필요하죠. `forge-release`가 그 다리를 놓습니다.

```
데이터 → 스코어러 top-k 추천 → 각 후보 풀 학습(early-stop, 40 epoch)
→ held-out test로 최고 선택 → model.keras + model_card.json 출력
```

**설계 원칙**: 스코어러는 **필터**(검색공간 576~6912 → 유망한 몇 개)이지 오라클이 아닙니다. 릴리스는 top-k를 **실제 학습으로 검증**하고 예측을 맹신하지 않습니다.

### 릴리스 라인업 (1.0)

각 릴리스는 스코어러 top-5를 풀 학습해 실측 최고를 선택. Image/Tabular는 v0.4 스코어러, TimeSeries는 v0.5, Text 1.2는 v0.6(bidirectional) 스코어러로 뽑음:

| 릴리스 | 데이터 | 실측 test | 우승 아키텍처 | 산출물 |
|---|---|---|---|---|
| **Lorenzo Image 1.0** | MNIST | **0.969** | 1× Conv2D(128, k5) adam | `releases/image_1_0/` |
| **Lorenzo Tabular 1.0** | `tabular_v1`(재현 가능, 4클래스) | **0.892** | 1× Dense(128, tanh) adam | `releases/tabular_1_0/` |
| **Lorenzo Text 1.2** | IMDB 감성(2클래스) | **0.836** | embed64 → bigru(32) adam | `releases/text_1_2/` |
| **Lorenzo Text 1.2 (Reuters)** | Reuters 토픽(46클래스) | **0.809** | embed64 → bigru(32) adam | `releases/text_reuters_1_2/` |
| **Lorenzo TimeSeries 1.0** | `timeseries_v1`(재현 가능, 5클래스) | **0.838** | 2× Conv1D(128, k5) adam | `releases/timeseries_1_0/` |

Text 1.2는 v0.6 bidirectional 스코어러로 우승이 bigru로 바뀌었으나 정확도는 1.1과 사실상 동일(위 v0.6 기록 참조).

텍스트는 1.0 대비 학습 예산(데이터·epoch)을 늘려 1.1로 갱신: IMDB 0.822 → **0.838**, Reuters 0.791 → **0.803**. 검색공간·스코어러는 그대로라 1.0 재현성은 유지되며, 개선은 실질적이지만 완만함(현 conv1d 계열의 천장 근처).

**"필터지 오라클 아님"이 반복 입증됨**:
- Image 1.0: 스코어러 1순위 예측이 실측 최고가 아니었고, top-5 실측 검증으로 진짜 최고(0.969)를 선택.
- Text 1.0(IMDB): GRU 후보가 예측 0.926인데 **실측 0.519(찍기 수준)로 붕괴** → 실제 학습이 이를 걸러내 conv1d를 선택. 예측만 믿었으면 붕괴 모델을 낼 뻔.
- Text 1.1(Reuters): 5번째 후보가 실측 0.608로 낮았지만 top-5 검증으로 conv1d(0.803)를 선택 — 같은 안전망 반복.
- Tabular 1.0: **재현 가능한 `tabular_v1` 데이터셋**(고정 시드, `datasets.py`)으로 재생성 → 1× Dense(128, tanh) adam 우승(실측 0.892, 4.5K params). 예전엔 휘발성 npz라 재현 불가였음.
- TimeSeries 1.0: 예측이 5개 후보 모두 ~0.998로 비슷했지만 **실측은 0.588~0.838로 크게 갈림** — 같은 2×conv1d라도 lr=0.01 후보는 0.588로 붕괴. top-5 실측으로 lr=1e-3(0.838)을 선택. `timeseries_v1`(고정 시드) 재현 가능.
- Forge의 값어치: 검색공간 전체가 아닌 **5개만 실제 학습**해도 최적 근접 → 탐색 비용 100배+ 절감.

산출물: `model.keras`(가중치) + `model_card.json`(데이터 프로파일 · 후보 전부의 예측/val/test · 우승 명세 · 재현정보). 릴리스 가중치(`model.keras`)는 대용량이라 gitignore, 카드는 저장소에 기록. 학습된 스코어러(`artifacts/scorer_model.keras`)와 코퍼스(`data/meta_training_corpus.jsonl`)는 크기가 작아 저장소에 커밋되어 있어, 새로 클론해도 `build-corpus`/`train-meta` 없이 바로 `release`를 이어갈 수 있음.

## 프로젝트 구조

```
lorenzo_forge/
  profile.py            # DataProfile: 데이터 특징 벡터 정의
  search_space.py        # ArchitectureSpec: 아키텍처 인코딩/열거/디코딩
  synthetic.py            # 합성 tabular 데이터 + (구) 합성 이미지 생성
  real_image.py           # 실제 이미지 프로파일 생성 (MNIST / Fashion-MNIST)
  text_data.py            # 실제 텍스트 프로파일 생성 (IMDB / Reuters)
  timeseries_data.py      # 합성 시계열 프로파일 생성 (주파수 + 국소 모티프)
  datasets.py             # 재현 가능한 릴리스용 고정 데이터셋 (tabular_v1 / timeseries_v1)
  candidate_trainer.py    # 후보 아키텍처를 실제로 빌드/학습/평가 (tabular/image/text)
  search.py               # 프로파일별 random search + 동률 tie-break
  dataset_builder.py       # tabular(합성)/image·text(실제) 3분할, 후보 전부 점수 기록
  meta_model.py            # 학습되는 스코어러 신경망 + 열거·점수화 추천 함수
  release.py               # Phase 1: top-k 풀 학습 → 최고 선택 → 모델+카드 패키징
  cli.py                   # lorenzo-forge CLI (build-corpus/train-meta/recommend/release)
tests_forge/               # 빠른 파라미터로 전체 파이프라인 검증
```

## 비목표

- 실행 가능한 학습 코드 자동 생성 (구조 명세 + 학습된 가중치까지만 출력)
- weight-sharing supernet, 진화 탐색 등 고급 NAS 기법 (작은 검색공간엔 과함 — README 상단 근거 참조)
- Transformer/attention 기반 대형 아키텍처 (현재 텍스트는 RNN/Conv1D 계열)
