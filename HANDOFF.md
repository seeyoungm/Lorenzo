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
| Lorenzo Tabular 1.1 | 표형(`tabular_v1`) | **0.922** (was 0.892) | `releases/tabular_1_1/` |
| Lorenzo TimeSeries 1.2 | 시계열(`timeseries_v1`) | **0.880** (was 0.867) | `releases/timeseries_1_2/` |
| Lorenzo Text 1.3 | 텍스트(IMDB) | 0.867 | `releases/text_1_3/` |
| Lorenzo Text 1.3 (Reuters) | 텍스트(Reuters 46클래스) | 0.818 | `releases/text_reuters_1_3/` |

(구버전 카드도 `releases/`에 히스토리로 남아 있음. 가중치 `model.keras`는 gitignore, 카드만 커밋.)

## 스코어러 상태 (v0.7)
- 4개 도메인(tabular/image/text/timeseries), 입력 42차원(v0.6의 38 + Track 2에서 추가된 `block_style`/`pool_style` 축 4차원).
- 인코더 축 5종: lstm/gru/conv1d/bilstm/bigru(bidirectional). **block_style**(plain/residual) · **pool_style**(standard/gap) 축이 tabular Dense·image Conv2D·conv1d-인코딩된 text/timeseries에 적용됨(재귀 인코더엔 미적용 — BatchNorm도 마찬가지로 재귀 레이어엔 안 붙임).
- 코퍼스 120 프로파일(v0.6과 동일 규모, CIFAR 없이 4도메인)로 재빌드 → 재학습.

## 어디까지 했고, 다음에 뭘 할지
목표: **각 도메인 릴리스 성능 개선.** 두 트랙:

**트랙 1 — 릴리스 학습 강화 (재빌드 불필요, 빠름) ← 완료**
- `release.full_train`에 LR 스케줄링(ReduceLROnPlateau) + patience 8 이미 적용됨.
- **Image 1.1**: 전체 MNIST + 강화학습으로 0.969→**0.990**.
- **Text 1.3 (IMDB)**: 0.836→**0.867** (`--num-samples 50000 --top-k 5 --epochs 40`).
- **Text 1.3 (Reuters)**: 0.809→**0.818** (`--num-samples 11228 --top-k 5 --epochs 40`). scorer의 predicted_accuracy가 46클래스 도메인에서 전부 0.000으로 나오는 기존 이슈 있음(순위 자체는 tie-break로 정상 작동해서 릴리스엔 지장 없었음) — 스코어러 쪽 별도 조사 필요.
- **TimeSeries 1.1**: 0.838→**0.867**. `--top-k 5`로 먼저 돌렸을 때는 오히려 0.739로 후퇴했다 — scorer의 predicted_accuracy가 top-12 전부 0.9996~0.9998로 사실상 동률이라, 근소한 순위 차이로 kernel=5 계열(실측 좋음)이 top-5에서 밀려나고 kernel=3 계열(실측 나쁨)만 뽑혔던 것. `--top-k 12`로 넓혀서 재검증하니 kernel=5/units=256 2블록이 진짜 승자(0.867)로 나옴. **교훈: predicted_accuracy가 촘촘히 몰려있는 도메인은 top-k를 늘려서 실측 검증 폭을 넓혀야 함** — release()가 "필터일 뿐 오라클 아님" 설계 원칙대로 동작하려면 필터 자체가 충분히 넓어야 함.
- **Tabular**: `--domain tabular_v1 --top-k 10 --epochs 60`으로 재릴리스 시도했으나 실측 0.658로 기존 1.0(0.892)보다 훨씬 나빠서 **채택하지 않음, 1.0 유지**. 원인: tabular_1_0(2026-07-07 13:46)은 그 이후 두 차례 스코어러 재학습(v0.5 4도메인 확장, v0.6 bidirectional)을 거치지 않은 구버전 스코어러로 뽑혔음. 현재 v0.6 스코어러로 다시 추천을 뽑아보면 원래 승자였던 tanh-128 아키텍처가 top-30 밖(predicted ~0.55대)으로 완전히 밀려나 있음 — relu 계열을 훨씬 선호하도록 랭킹이 바뀌었는데, 정작 relu 계열은 실측하면 전부 0.6대에 그침. 스코어러를 재학습할 때마다 특정 도메인의 예전 최적점을 못 찾을 수 있다는 뜻 — top-k를 아무리 넓혀도 (top-30까지 확인함) 해당 아키텍처 자체가 후보에 안 들어가면 소용없음. 재도전하려면 스코어러 재학습이 아니라 **아키텍처 자체를 직접 지정**(`--data-npz` 경로나 코드로 직접 tanh-128을 학습)해서 실측 비교하는 게 정공법.

**트랙 2 — 검색공간 현대화 ← 완료 (CIFAR 제외)**
- `search_space.py`에 `block_style`(plain/residual) · `pool_style`(standard/gap) 축 추가. `candidate_trainer.py`: 비-재귀 블록(tabular Dense, image Conv2D, conv1d 인코더)엔 전부 BatchNorm 무조건 적용(`Linear/Conv(use_bias=False) -> BN -> Activation`), block_style이 residual이면 2-conv/dense + skip(채널 다르면 1x1/1-unit projection), pool_style이 gap이면 image는 `GlobalAveragePooling2D`(기존 Flatten+Dense 대신), conv1d 인코더는 `GlobalAveragePooling1D`(기존 GlobalMaxPooling1D 대신). 재귀 인코더(lstm/gru/bilstm/bigru)는 그대로(BN/residual 비표준이라 범위 밖).
- **CIFAR-10 도메인은 이번엔 제외**: 공식 배포처 `cs.toronto.edu`가 이 네트워크에서 초당 ~500바이트(170MB 파일 기준 80시간+)로 사실상 다운로드 불가. `tensorflow-datasets`의 `try_gcs=True`도 시도했지만 GCS 사전빌드본을 못 찾고 결국 같은 느린 호스트로 연결됨(cifar10 3.0.2가 GCS에 안 올라가 있거나 이 tfds 버전에서 못 찾는 듯). `real_image.py`의 `SOURCES`/`release.py`의 `BUILTIN_IMAGE_DOMAINS`에 `cifar10`을 추가했다가 이 문제로 되돌림 — 코드에 남아있지 않음. **재도전하려면**: (a) 다른 네트워크 경로 확보, (b) `cifar-10-python.tar.gz`를 어디선가 받아서 `~/.keras/datasets/`에 직접 넣어두고 시작(케라스 로더가 캐시를 그대로 씀), (c) tensorflow-datasets를 쓰려면 `pip install tensorflow-metadata==1.16.1 importlib_resources`까지 같이 깔아야 protobuf 버전 충돌(tensorflow<2.19가 요구하는 protobuf<6 vs tensorflow-metadata 최신판이 요구하는 protobuf>=6.31) 없이 임포트됨 — 그래도 GCS 미러가 실제로 존재하는지는 별개 문제.
- **재릴리스 결과** (top-k를 5~12로 넓혀 실측 검증, 트랙 1 교훈 적용):
  - **Tabular 1.1: 0.892→0.922** ✓ (`--top-k 10 --epochs 60`, 승자 `1x Dense(units=128, act=tanh)`, block_style/pool_style 특별히 안 씀)
  - **TimeSeries 1.2: 0.867→0.880** ✓ (`--top-k 12 --epochs 80`, 승자는 `1x bilstm(units=256)` — 재귀 인코더라 모더나이제이션 축 자체는 무관, top-k를 넓혀서 더 좋은 재귀 후보를 찾은 게 개선의 핵심)
  - **Image: 0.990 시도 → 0.983로 후퇴, 채택 안 함** (`--top-k 10 --epochs 60`, 풀 MNIST 7만 샘플). predicted_accuracy가 top-10 전부 0.846~0.862로 촘촘히 몰려 num_blocks=1 계열만 뽑혔고, residual 후보(2블록 254유닛 5x5커널)는 후보당 50분 넘게 걸려 top-k를 더 넓히는 비용이 너무 큼 (이 릴리스 하나에만 총 ~3시간 소요). 기존 1.1 유지.
  - **Text (IMDB): 0.867 시도 → 0.861로 후퇴, 채택 안 함.** **Text (Reuters): 0.818 시도 → 0.794로 후퇴, 채택 안 함.** 둘 다 top-10 후보 중 일부(bigru 계열)가 실측 시 거의 랜덤 수준(val/test ≈ 0.000~0.5)으로 붕괴 — Track 1에서 관찰됐던 "GRU 계열 예측은 높은데 실측 붕괴" 패턴이 재현됨. 이 텍스트 도메인 학습 불안정성은 Track 2가 만든 문제가 아니라 기존부터 있던 것으로 보이고, 별도 조사가 필요함(아래 "다음 작업" 참고).
  - **결론**: 스코어러를 재학습할 때마다 도메인별로 이전 최적점을 못 찾는 "스코어러 드리프트" 현상이 tabular(트랙 1) 이후 image/text/text-reuters(트랙 2)에서도 반복 관찰됨. top-k를 넓히는 것만으로는 항상 해결되지 않음(특히 image처럼 후보당 비용이 큰 도메인) — `release()`가 "필터일 뿐, 실측이 최종 결정"이라는 설계 원칙 덕분에 이번에도 나쁜 결과를 자동으로 걸러낼 수 있었음.
- **엔지니어링 사고 두 건 (둘 다 수정 완료)**:
  1. **메모리 누수**: 코퍼스 빌드가 프로파일당 후보 여러 개 × 120 프로파일 = 수백 개 모델을 한 프로세스 안에서 연달아 만드는데, `tf.keras.backend.clear_session()`을 안 불러서 Keras 전역 레이어/uid 레지스트리와 tf.function 리트레이싱 캐시가 계속 쌓여 RSS가 21GB+까지 치솟으며 계속 증가했음. `candidate_trainer.evaluate_spec()`의 `finally` 블록에 `clear_session()` 추가로 해결(속도는 약간 손해 — 매 후보마다 그래프 재구성 필요하지만 안정성이 우선).
  2. **장시간 백그라운드 프로세스가 이 환경에서 불안정함**: 120 프로파일을 한 번에(nohup, 청크 없이) 돌렸더니 ~3시간 경과 후 74/120에서 아무 에러 로그·OOM 흔적·절전 이벤트 없이 조용히 죽었음(원인 특정 못 함). `build_training_corpus`는 **루프가 끝까지 돌아야만** `out_path`에 씀 — 74개 진행분이 전부 날아감. **대응**: 120개를 4개 청크(각 `--profiles 30`, `--seed 0/1/2/3` 다르게, 각자 다른 파일로) 로 나눠 순차 실행 후 `cat`으로 합침. 청크 하나가 죽어도 최대 30개치만 손실. 절전 가능성을 배제하려고 각 청크를 `caffeinate -i`로 감쌈(원인이었는지는 불확실하지만 비용 없는 보험).
- 텍스트 정확도 천장(현 ~0.86)을 진짜 뚫으려면 사전학습 임베딩(GloVe) 또는 Transformer 필요 — 비용 큼, 이번 트랙 2 범위에서 명시적으로 제외.

**엔지니어링 개선 후보**
- **증분 코퍼스 — 구현 완료**: `build_training_corpus(..., base_corpus_path=...)` / CLI `--base-corpus` + `--domains`. `domains`에 지정한 도메인만 새로 서치하고, 나머지 도메인 레코드는 base corpus에서 그대로 이어받는다.
  ```bash
  lorenzo-forge build-corpus --profiles 30 --domains text --base-corpus lorenzo_forge/data/meta_training_corpus.jsonl \
      --out lorenzo_forge/data/meta_training_corpus.jsonl   # text만 재서치, tabular/image/timeseries는 유지
  ```
  주의: `--out`을 base와 같은 경로로 지정하면 그 자리에서 갱신됨(먼저 읽고 다 쓴 뒤 저장하므로 안전). 검색공간을 바꾼 경우엔 이 기능 무의미 — 전체 도메인 재빌드 필요(위 "함정" 참고).

## 코퍼스/스코어러 재빌드 방법
검색공간(HEADS)이나 프로파일 특징을 바꿨다면 스코어러 입력 차원이 바뀌므로 반드시 재빌드+재학습. **120개를 한 번에 돌리지 말 것** — 위 "엔지니어링 사고" 참고, 청크로 나눠서:
```bash
S=lorenzo_forge/data/meta_training_corpus.jsonl
for seed in 0 1 2 3; do
  caffeinate -i lorenzo-forge build-corpus --profiles 30 --candidates 8 --search-epochs 4 --seed $seed \
      --out /tmp/corpus_chunk_$seed.jsonl
done
cat /tmp/corpus_chunk_*.jsonl > $S
lorenzo-forge train-meta --epochs 30 --corpus $S --out lorenzo_forge/artifacts/scorer_model.keras
```
청크 하나가 몇 시간 걸릴 수 있음(로컬 M3 GPU라고 항상 빠르진 않음 — 아래 "함정" 참고). 매 청크 끝나면 `wc -l`로 개수 확인하고 다음 청크로.

## 함정 / 주의
- 릴리스 가중치(`model.keras`)와 대용량은 gitignore. **스코어러·코퍼스·카드는 커밋**. 저장소에 남길 카드는 `git add -f`.
- 검색공간(search_space.py의 HEADS/인코더)이나 프로파일 특징(profile.py FEATURE_NAMES)을 바꾸면 **스코어러 입력 차원이 바뀌어** 기존 스코어러 못 씀 → 재빌드 필수.
- 합성 도메인(tabular_v1/timeseries_v1)은 노이즈를 우리가 고정 → 정확도 상한이 인위적. 재현성은 `datasets.py`의 고정 시드로 보장.
- 코퍼스 빌드 단위테스트는 `domains=("tabular",)`로 hermetic 고정(네트워크/느린 텍스트 회피).
- 원격이 **git 태그 push를 거부**함(브랜치만 됨). memory-centric 옛 코드는 커밋 `a9ac432`에 보존(`git checkout a9ac432 -- lorenzo`).
- **M3 Max GPU(Metal)가 이 워크로드엔 항상 유리하지 않음**: 코퍼스 빌드/서치는 유닛 16~256개짜리 작은 모델을 수백 개 학습하는 건데, tensorflow-metal은 연산을 GPU로 디스패치하는 오버헤드가 작은 텐서에선 실제 연산 시간보다 커질 수 있음. 게다가 LSTM/GRU/BiLSTM/BiGRU는 시간축 순차 처리 특성상 어떤 백엔드에서도 느림. "GPU니까 빠를 것"이라고 낙관하지 말 것 — 실측 소요 시간을 로그로 확인하며 진행.
- **코퍼스 빌드는 `tf.keras.backend.clear_session()` 없이 오래 돌리면 메모리가 무한정 늘어남**(RSS 21GB+까지 확인). `candidate_trainer.evaluate_spec()`에 이미 수정됨 — 새로 비슷한 루프를 짤 때도 같은 패턴 조심.
- **장시간(1시간+) 백그라운드 프로세스가 이 환경에서 이유 없이 죽을 수 있음**(3시간짜리 코퍼스 빌드가 74/120에서 조용히 종료, 에러 로그 없음). 결과를 끝에만 쓰는 함수는 반드시 청크로 나눠 중간 산출물을 남길 것.
- CIFAR-10 원본 배포처(`cs.toronto.edu`)가 이 네트워크에서 극단적으로 느림(~500B/s) — tensorflow-datasets GCS 미러(`try_gcs=True`)도 우회 안 됨. CIFAR 재도전 시 이 문서의 트랙 2 섹션 참고.

## 주요 파일
`search_space.py`(아키텍처 인코딩/열거) · `profile.py`(데이터 특징) · `candidate_trainer.py`(모델 빌드/학습, 4도메인+bidirectional) · `search.py`(랜덤서치+tie-break) · `dataset_builder.py`(코퍼스) · `meta_model.py`(스코어러) · `release.py`(릴리스+full_train) · `datasets.py`(재현 데이터) · `real_image.py`/`text_data.py`/`timeseries_data.py`(도메인 데이터) · `cli.py`

## 다음 로컬 작업 제안 (트랙 1·2 완료 후)
1. **텍스트 도메인 학습 불안정성 조사**: IMDB/Reuters 재릴리스 후보 중 bigru/lr=0.01 근방 조합이 종종 val/test ≈ 0.000~0.5로 붕괴. 그래디언트 폭주인지 특정 lr 값 문제인지 확인해볼 만함 — 고쳐지면 top-k를 더 좁게 잡아도 안전해져서 텍스트 재릴리스 비용이 줄어듦.
2. **CIFAR-10**: 네트워크 문제로 보류. 다른 경로로 `cifar-10-python.tar.gz`를 `~/.keras/datasets/`에 받아두고 재시도, 또는 다른 네트워크에서.
3. Reuters 스코어러 predicted_accuracy=0.000 이슈 조사 (여전히 미해결, v0.7에서도 재현됨).
4. 사전학습 임베딩(GloVe)/Transformer로 텍스트 정확도 천장(~0.86) 돌파 — 비용 큰 별도 이니셔티브.
5. Image 도메인은 predicted_accuracy가 매우 촘촘(0.846~0.862대)해서 num_blocks=1 계열만 계속 뽑힘 — residual 후보가 후보당 50분+ 걸려 top-k를 넓히기 부담스러움. 검색공간에서 image의 residual 변형을 유닛 수가 작을 때만 시도하게 제한하면(예: units<=128) 더 넓게 탐색 가능할 수도.
