# Lorenzo

**Lorenzo는 도메인별로 최적화된 신경망 릴리스 라인입니다.** 각 릴리스는 데이터 도메인(이미지·표형·텍스트·시계열)에 맞춰 아키텍처를 탐색·검증한 뒤, 실제로 학습·패키징한 모델입니다.

이 릴리스들을 만들어내는 엔진이 **[Lorenzo Forge](lorenzo_forge/README.md)** — 데이터 프로파일을 보고 좋은 신경망 아키텍처를 추천하는 학습된 메타모델(predictor-based NAS)입니다.

> Lorenzo = 제품(릴리스), Forge = 그 제품을 찍어내는 도구.

## 릴리스 카탈로그

각 릴리스는 Forge 스코어러의 top-5 추천을 **실제로 끝까지 학습**해 실측 최고를 고른 결과입니다. 산출물은 `lorenzo_forge/releases/<name>/`의 `model_card.json`(데이터 프로파일·후보 전부의 예측/실측·우승 명세·재현정보)이며, 가중치(`model.keras`)는 CLI로 재생성됩니다.

| 릴리스 | 도메인 | 데이터 | 실측 test acc | 우승 아키텍처 |
|---|---|---|---|---|
| **Lorenzo Image 1.0** | 이미지 | MNIST | **0.969** | 1× Conv2D(128, k5) adam |
| **Lorenzo Tabular 1.0** | 표형 | `tabular_v1` (재현 가능) | **0.892** | 1× Dense(128, tanh) adam |
| **Lorenzo TimeSeries 1.0** | 시계열 | `timeseries_v1` (재현 가능) | **0.838** | 2× Conv1D(128, k5) adam |
| **Lorenzo Text 1.2** | 텍스트 | IMDB 감성(2클래스) | **0.836** | embed64 → bigru(32) adam |
| **Lorenzo Text 1.2 (Reuters)** | 텍스트 | Reuters 토픽(46클래스) | **0.809** | embed64 → bigru(32) adam |

## 새 릴리스 만들기

```bash
cd lorenzo_forge
pip install -e ".[dev]"

# 내장 재현 가능 도메인으로 릴리스 (스코어러는 저장소에 커밋되어 있음)
lorenzo-forge release --name "Lorenzo Image 1.0" --domain mnist \
    --scorer lorenzo_forge/artifacts/scorer_model.keras \
    --top-k 5 --epochs 40 --out-dir lorenzo_forge/releases/image_1_0

# 도메인: mnist / fashion_mnist / imdb / reuters / tabular_v1 / timeseries_v1
# 커스텀 데이터: --data-npz X,y.npz --task {tabular,image,text,timeseries}
```

엔진의 동작 원리·실험 기록(v0.1 실패 → v0.6)·설계 결정은 **[lorenzo_forge/README.md](lorenzo_forge/README.md)** 참조.

## 설계 원칙

- **Forge는 필터, 릴리스는 실측 검증.** 스코어러는 검색공간(수백~1만 개)을 유망한 5개로 좁혀 탐색 비용을 100배+ 절감하지만, 오라클이 아닙니다. 릴리스는 top-5를 **실제로 학습해** 최고를 확정합니다(예측 상위가 실측 최고가 아닌 사례 다수).
- **재현성.** 릴리스 데이터는 고정 시드 내장 도메인(`tabular_v1`/`timeseries_v1`) 또는 공개 데이터셋(MNIST·Fashion·IMDB·Reuters). 스코어러·코퍼스도 저장소에 커밋되어 클론만으로 릴리스를 이어갈 수 있습니다.

## 저장소 구조

```
README.md            # 이 파일 — Lorenzo 릴리스 카탈로그
lorenzo_forge/       # Forge 엔진 (아키텍처 추천 + 릴리스 파이프라인)
  lorenzo_forge/     # 파이썬 패키지
  releases/          # Lorenzo 릴리스 산출물 (model_card.json)
  artifacts/         # 학습된 스코어러
  data/              # 메타 학습 코퍼스
  README.md          # 엔진 상세 문서
tests_forge/         # 엔진 테스트
```

## 이전 프로젝트 (memory-centric Lorenzo)

Lorenzo는 원래 memory-centric AI 프로토타입(v1~v2)이었고, 이후 **"도메인별 신경망을 생성·릴리스하는" 방향으로 피벗**했습니다. 이전 memory-centric 코드는 삭제되지 않고 **git 히스토리에 온전히 보존**되어 있습니다 — 원격 커밋 `a9ac432`(memory-centric 마지막 릴리스, 이전 `main` 상태)에서 복원할 수 있습니다:

```bash
git checkout a9ac432        # memory-centric Lorenzo (v1-v2) 트리 확인
git checkout a9ac432 -- lorenzo  # 또는 lorenzo/ 패키지만 꺼내기
```
