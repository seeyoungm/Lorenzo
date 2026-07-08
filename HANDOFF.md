# 인수인계 — Lorenzo Forge (로컬 M3 세션용)

이 문서는 맥북(M3 Pro) 로컬 터미널 Claude Code로 작업을 이어받기 위한 것입니다. 이전 작업은 클라우드(4-CPU, GPU 없음) 세션에서 진행됐고, 이제 로컬 GPU로 더 빠르게 이어갑니다.

## 프로젝트가 뭔지
- 저장소: `seeyoungm/lorenzo`. 브랜치 `claude/project-status-cpx9nb` == `main` (동기화됨). 최신 커밋 `59b5b0b`.
- **Lorenzo = 도메인별 신경망 릴리스 라인(제품)**, **Forge(`lorenzo_forge/`) = 그 릴리스를 만드는 엔진**(predictor-based NAS).
- 흐름: 데이터 프로파일 → 학습된 스코어러가 아키텍처 top-k 추천 → 각 후보를 실제로 풀 학습 → 실측 최고를 릴리스로 패키징.
- **설계 원칙**: 스코어러는 오라클이 아니라 필터. 반드시 top-k를 실측 학습해 확정(예측 1순위 ≠ 실측 최고 사례 다수, GRU가 예측 높은데 실측 붕괴한 사례도 있음).

## 로컬 셋업 (Apple Silicon)
```bash
git clone <repo> && cd Lorenzo          # main 또는 claude/project-status-cpx9nb
python3 -m venv .venv && source .venv/bin/activate
cd lorenzo_forge
pip install -e ".[dev,metal]"           # arm64: tensorflow(<2.19, pinned) + tensorflow-metal(GPU 가속)
python -m pytest ../tests_forge -q       # 전체 통과해야 정상
```
스코어러(`artifacts/scorer_model.keras`)와 코퍼스(`data/…jsonl`)가 커밋돼 있어 **클론 즉시 릴리스 가능**. CIFAR 등 클라우드에서 막히던 데이터셋도 로컬에선 받힘.

**GPU 관련 함정**: PyPI의 `tensorflow-metal`은 최신판(1.2.0)이 나온 뒤에도 `tensorflow` 쪽은 계속 새 버전이 나와서, `pip install`이 기본으로 최신 tensorflow(예: 2.21)를 잡으면 `import tensorflow` 자체가 `libmetal_plugin.dylib` 심볼 로드 실패로 죽는다(GPU 가속 상실 정도가 아니라 임포트 자체가 깨짐). `pyproject.toml`에 arm64용으로 `tensorflow<2.19` 상한을 걸어뒀다 — tensorflow 2.18.1 + tensorflow-metal 1.2.0 조합이 M3 Max에서 확인된 정상 조합. 만약 다시 깨지면 `pip index versions tensorflow-metal`로 새 버전이 나왔는지 확인하고 상한을 갱신할 것.

## 현재 릴리스 라인업 (실측 test acc)
| 릴리스 | 도메인 | test | 산출물 |
|---|---|---|---|
| Lorenzo Image 1.1 | 이미지(전체 MNIST) | **0.990** | `releases/image_1_1/` |
| Lorenzo Tabular 1.0 | 표형(`tabular_v1`) | 0.892 | `releases/tabular_1_0/` |
| Lorenzo TimeSeries 1.1 | 시계열(`timeseries_v1`) | **0.867** (was 0.838) | `releases/timeseries_1_1/` |
| Lorenzo Text 1.3 | 텍스트(IMDB) | **0.867** (was 0.836) | `releases/text_1_3/` |
| Lorenzo Text 1.3 (Reuters) | 텍스트(Reuters 46클래스) | **0.818** (was 0.809) | `releases/text_reuters_1_3/` |

(구버전 카드도 `releases/`에 히스토리로 남아 있음. 가중치 `model.keras`는 gitignore, 카드만 커밋.)

## 스코어러 상태 (v0.6)
- 4개 도메인(tabular/image/text/timeseries), 입력 38차원.
- 인코더 축 5종: lstm/gru/conv1d/**bilstm/bigru**(bidirectional). held-out Spearman: image 0.89 / tabular 0.78 / timeseries 0.75 / **text 0.58**(bidirectional로 0.40→0.58 개선) / overall 0.75.

## 어디까지 했고, 다음에 뭘 할지
목표: **각 도메인 릴리스 성능 개선.** 두 트랙:

**트랙 1 — 릴리스 학습 강화 (재빌드 불필요, 빠름) ← 완료**
- `release.full_train`에 LR 스케줄링(ReduceLROnPlateau) + patience 8 이미 적용됨.
- **Image 1.1**: 전체 MNIST + 강화학습으로 0.969→**0.990**.
- **Text 1.3 (IMDB)**: 0.836→**0.867** (`--num-samples 50000 --top-k 5 --epochs 40`).
- **Text 1.3 (Reuters)**: 0.809→**0.818** (`--num-samples 11228 --top-k 5 --epochs 40`). scorer의 predicted_accuracy가 46클래스 도메인에서 전부 0.000으로 나오는 기존 이슈 있음(순위 자체는 tie-break로 정상 작동해서 릴리스엔 지장 없었음) — 스코어러 쪽 별도 조사 필요.
- **TimeSeries 1.1**: 0.838→**0.867**. `--top-k 5`로 먼저 돌렸을 때는 오히려 0.739로 후퇴했다 — scorer의 predicted_accuracy가 top-12 전부 0.9996~0.9998로 사실상 동률이라, 근소한 순위 차이로 kernel=5 계열(실측 좋음)이 top-5에서 밀려나고 kernel=3 계열(실측 나쁨)만 뽑혔던 것. `--top-k 12`로 넓혀서 재검증하니 kernel=5/units=256 2블록이 진짜 승자(0.867)로 나옴. **교훈: predicted_accuracy가 촘촘히 몰려있는 도메인은 top-k를 늘려서 실측 검증 폭을 넓혀야 함** — release()가 "필터일 뿐 오라클 아님" 설계 원칙대로 동작하려면 필터 자체가 충분히 넓어야 함.
- **Tabular**: `--domain tabular_v1 --top-k 10 --epochs 60`으로 재릴리스 시도했으나 실측 0.658로 기존 1.0(0.892)보다 훨씬 나빠서 **채택하지 않음, 1.0 유지**. 원인: tabular_1_0(2026-07-07 13:46)은 그 이후 두 차례 스코어러 재학습(v0.5 4도메인 확장, v0.6 bidirectional)을 거치지 않은 구버전 스코어러로 뽑혔음. 현재 v0.6 스코어러로 다시 추천을 뽑아보면 원래 승자였던 tanh-128 아키텍처가 top-30 밖(predicted ~0.55대)으로 완전히 밀려나 있음 — relu 계열을 훨씬 선호하도록 랭킹이 바뀌었는데, 정작 relu 계열은 실측하면 전부 0.6대에 그침. 스코어러를 재학습할 때마다 특정 도메인의 예전 최적점을 못 찾을 수 있다는 뜻 — top-k를 아무리 넓혀도 (top-30까지 확인함) 해당 아키텍처 자체가 후보에 안 들어가면 소용없음. 재도전하려면 스코어러 재학습이 아니라 **아키텍처 자체를 직접 지정**(`--data-npz` 경로나 코드로 직접 tanh-128을 학습)해서 실측 비교하는 게 정공법.

**트랙 2 — 검색공간 현대화 (재빌드 필요, but M3에선 빠름)**
- BatchNorm / residual conv 블록 / global-avg-pooling 추가 → 천장 상향(모든 도메인).
- 텍스트 정확도 천장(현 ~0.84)을 진짜 뚫으려면 사전학습 임베딩(GloVe) 또는 Transformer 필요 — 비용 큼, 로컬에서 검토.
- CIFAR(컬러 이미지) 도메인 추가도 로컬에선 가능.

**엔지니어링 개선 후보**
- **증분 코퍼스**: 지금은 도메인 하나 바꿔도 120 프로파일 전체 재빌드(느림). 기존 코퍼스 재사용 + 바뀐 도메인만 추가하도록 `dataset_builder`에 기능 추가하면 재빌드가 훨씬 싸짐. (아직 미구현 — 좋은 첫 로컬 작업)

## 코퍼스/스코어러 재빌드 방법 (트랙 2에 필요)
```bash
lorenzo-forge build-corpus --profiles 120 --candidates 6 --search-epochs 4   # 4-도메인, 몇 시간(로컬 GPU면 단축)
lorenzo-forge train-meta --epochs 30
# 검색공간을 바꿨다면 스코어러 입력 차원이 바뀌므로 반드시 재빌드+재학습
```

## 함정 / 주의
- 릴리스 가중치(`model.keras`)와 대용량은 gitignore. **스코어러·코퍼스·카드는 커밋**. 저장소에 남길 카드는 `git add -f`.
- 검색공간(search_space.py의 HEADS/인코더)이나 프로파일 특징(profile.py FEATURE_NAMES)을 바꾸면 **스코어러 입력 차원이 바뀌어** 기존 스코어러 못 씀 → 재빌드 필수.
- 합성 도메인(tabular_v1/timeseries_v1)은 노이즈를 우리가 고정 → 정확도 상한이 인위적. 재현성은 `datasets.py`의 고정 시드로 보장.
- 코퍼스 빌드 단위테스트는 `domains=("tabular",)`로 hermetic 고정(네트워크/느린 텍스트 회피).
- 원격이 **git 태그 push를 거부**함(브랜치만 됨). memory-centric 옛 코드는 커밋 `a9ac432`에 보존(`git checkout a9ac432 -- lorenzo`).

## 주요 파일
`search_space.py`(아키텍처 인코딩/열거) · `profile.py`(데이터 특징) · `candidate_trainer.py`(모델 빌드/학습, 4도메인+bidirectional) · `search.py`(랜덤서치+tie-break) · `dataset_builder.py`(코퍼스) · `meta_model.py`(스코어러) · `release.py`(릴리스+full_train) · `datasets.py`(재현 데이터) · `real_image.py`/`text_data.py`/`timeseries_data.py`(도메인 데이터) · `cli.py`

## 다음 로컬 작업 제안 (트랙 1 완료 후)
1. 증분 코퍼스 기능(`dataset_builder`) — 아직 미구현, 좋은 다음 작업.
2. 트랙 2(BatchNorm/residual/GAP) or CIFAR 도메인 — 재빌드 필요, 시간 크게 듦.
3. Reuters 스코어러 predicted_accuracy=0.000 이슈 조사 (위 참고).
