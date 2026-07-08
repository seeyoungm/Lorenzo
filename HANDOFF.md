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
pip install -e ".[dev,metal]"           # arm64: tensorflow + tensorflow-metal(GPU 가속)
python -m pytest ../tests_forge -q       # 전체 통과해야 정상
```
스코어러(`artifacts/scorer_model.keras`)와 코퍼스(`data/…jsonl`)가 커밋돼 있어 **클론 즉시 릴리스 가능**. CIFAR 등 클라우드에서 막히던 데이터셋도 로컬에선 받힘.

## 현재 릴리스 라인업 (실측 test acc)
| 릴리스 | 도메인 | test | 산출물 |
|---|---|---|---|
| Lorenzo Image 1.1 | 이미지(전체 MNIST) | **0.990** | `releases/image_1_1/` |
| Lorenzo Tabular 1.0 | 표형(`tabular_v1`) | 0.892 | `releases/tabular_1_0/` |
| Lorenzo TimeSeries 1.0 | 시계열(`timeseries_v1`) | 0.838 | `releases/timeseries_1_0/` |
| Lorenzo Text 1.2 | 텍스트(IMDB) | 0.836 | `releases/text_1_2/` |
| Lorenzo Text 1.2 (Reuters) | 텍스트(Reuters 46클래스) | 0.809 | `releases/text_reuters_1_2/` |

(구버전 카드도 `releases/`에 히스토리로 남아 있음. 가중치 `model.keras`는 gitignore, 카드만 커밋.)

## 스코어러 상태 (v0.6)
- 4개 도메인(tabular/image/text/timeseries), 입력 38차원.
- 인코더 축 5종: lstm/gru/conv1d/**bilstm/bigru**(bidirectional). held-out Spearman: image 0.89 / tabular 0.78 / timeseries 0.75 / **text 0.58**(bidirectional로 0.40→0.58 개선) / overall 0.75.

## 어디까지 했고, 다음에 뭘 할지
목표: **각 도메인 릴리스 성능 개선.** 두 트랙:

**트랙 1 — 릴리스 학습 강화 (재빌드 불필요, 빠름) ← 진행 중**
- `release.full_train`에 LR 스케줄링(ReduceLROnPlateau) + patience 8 이미 적용됨.
- **Image 1.1 완료**: 전체 MNIST + 강화학습으로 0.969→**0.990**. 트랙1 유효성 입증.
- **남은 일 (로컬에서 바로)**: 같은 방식으로 나머지 재릴리스
  ```bash
  S=lorenzo_forge/artifacts/scorer_model.keras
  lorenzo-forge release --name "Lorenzo Text 1.3" --domain imdb --scorer $S \
      --num-samples 50000 --top-k 5 --epochs 40 --out-dir lorenzo_forge/releases/text_1_3
  lorenzo-forge release --name "Lorenzo Text 1.3 (Reuters)" --domain reuters --scorer $S \
      --num-samples 11228 --top-k 5 --epochs 40 --out-dir lorenzo_forge/releases/text_reuters_1_3
  lorenzo-forge release --name "Lorenzo TimeSeries 1.1" --domain timeseries_v1 --scorer $S \
      --top-k 5 --epochs 80 --out-dir lorenzo_forge/releases/timeseries_1_1
  # Tabular도 여지 있으면 --domain tabular_v1로 재릴리스
  ```

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

## 첫 로컬 작업 제안
1. 클론 → `pip install -e ".[dev,metal]"` → `pytest tests_forge -q` 통과 확인
2. 트랙 1 나머지 도메인 재릴리스(위 명령) → 성능 표 갱신 → 커밋/푸시
3. 여유되면 증분 코퍼스 기능 or 트랙 2(BatchNorm/residual) or CIFAR 도메인
