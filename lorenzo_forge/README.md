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
