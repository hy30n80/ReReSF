# Retriever 학습 및 추론 시스템

이 시스템은 SFR(Salesforce Foundation Retrieval) 모델을 사용하여 Text-to-Cypher 검색을 위한 Dense Passage Retrieval 모델을 학습하고 추론하는 코드입니다.

## 📁 시스템 구조

```
2.Retrieval/
├── train_retriever.py    # 모델 학습
├── run_retriever.py      # 모델 추론 및 Cross-Encoder 데이터 생성
├── eval.py              # 모델 성능 평가
├── sfr_models/          # 학습된 모델 저장 디렉토리
├── results/             # 평가 결과 저장 디렉토리
├── models/              # 추가 모델 저장 디렉토리
├── train.sh             # 학습 실행 스크립트
├── run.sh               # 추론 실행 스크립트
└── README.md           # 이 파일
```

## 🚀 빠른 시작

### 1. 스크립트 사용 (내가 CypherBench 에서 사용했던 Bash Shell)

```bash
# Train 학습 스크립트
./train.sh 

# Inference 추론 스크립트 (학습된 모델로, Train-set, Test-set 의 Top-20 결과 추출하기) 
./run.sh
```


### 2. 모델 학습 (Custom)

```bash
# 커스텀 설정으로 학습 (예시)
python train_retriever.py \
    --train_file_path ../datasets/cypherbench/train.json \
    --model_path ./models/my_model \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --result_path my_results
```

### 3. 모델 추론 (Custom)

```bash
# 커스텀 설정으로 추론  (예시)
python run_retriever.py \
    --train_file_path ../datasets/cypherbench/train.json \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 2e-5 \
    --retriever my_model-ep15
```


## ⚙️ 파라미터 설명

### train_retriever.py 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--train_file_path` | `../datasets/cypherbench/train.json` | 학습 데이터 파일 경로 |
| `--model_path` | `./models/SFR` | 모델 저장 경로 |
| `--epochs` | `21` | 학습 에포크 수 |
| `--batch_size` | `32` | 배치 크기 |
| `--learning_rate` | `2e-5` | 학습률 |
| `--result_path` | `SFR` | 결과 파일 이름 |

### run_retriever.py 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--train_file_path` | `../datasets/cypherbench/train.json` | 학습 데이터 파일 경로 |
| `--epochs` | `30` | 에포크 수 |
| `--batch_size` | `128` | 배치 크기 |
| `--learning_rate` | `2e-5` | 학습률 |
| `--retriever` | `sfr2-ep15` | 사용할 모델명 |

## 📋 데이터 형식

### 입력 데이터 형식 (JSON 배열)

데이터는 JSON 배열 형식이어야 하며, 각 객체는 다음과 같은 필드를 포함해야 합니다:

```json
[
  {
    "qid": "unique_id",
    "graph": "graph_name",
    "gold_cypher": "original_cypher_query",
    "nl_question": "질문 텍스트",
    "masked_cypher": "마스킹된 Cypher 쿼리",
    "from_template": {...},
    "masked_question": "마스킹된 질문",
    "negative_cypher": [...],
    "hard_negative_cypher": [...]
  }
]
```

### 필수 필드

- `nl_question`: 자연어 질문
- `masked_cypher`: 마스킹된 Cypher 쿼리 (학습에 사용)
- `graph`: 그래프 이름

### 출력 데이터

- **모델 파일**: `models/{model_name}/query_enc_model/`, `sql_enc_model/`
- **모델 성능 결과**: `results/{model_name}_seen.csv`, `results/{model_name}_unseen.csv`
- **Dataset 의 Train NLQ / Test NLQ 에 대해서 학습된 모델로 Top-20 추출한 결과**: `../outputs/cypherbench/retrieval_results/{model_name}/train.json`, `test.json`

## 📈 평가 메트릭

시스템은 다음 메트릭들을 제공합니다:

- **Recall@1, Recall@5, Recall@10, Recall@20**: Top-K 정확도
- **MRR (Mean Reciprocal Rank)**: 평균 역순위
- **Mean Rank, Max Rank**: 평균/최대 순위
- **Average Test Loss**: 평균 테스트 손실

