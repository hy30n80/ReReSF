# Re-ranker 학습 및 추론 시스템

이 시스템은 SFR(Salesforce Foundation Retrieval) 모델을 사용하여 Text-to-Cypher 검색을 위한 Cross-Encoder Re-ranker 모델을 학습하고 추론하는 코드입니다.

## 📁 시스템 구조

```
3.Re-ranking/
├── train_reranker.py    # Re-ranker 모델 학습
├── run_reranker.py      # Re-ranker 모델 추론
├── train.sh             # 학습 실행 스크립트
├── run.sh               # 추론 실행 스크립트
├── models/              # 학습된 모델 저장 디렉토리
├── results/             # 평가 결과 저장 디렉토리
└── README.md           # 이 파일
```

## 🚀 빠른 시작

### 1. 스크립트 사용 (내가 CypherBench 에서 사용했던 Bash Shell)

```bash
# Train 학습 스크립트
./train.sh 

# Inference 추론 스크립트 (학습된 모델로, Train-set, Test-set 의 Top-20 결과 재정렬하기) 
./run.sh
```

### 2. Re-ranker 모델 학습 (Custom)

#### 인자 설명 
- model_path : 학습된 Reranker 저장할 경로
- retrieval_results_path : 이전 단계인 Retriever 을 통해 생성한 (NLQ, Top-20 Candidates) 폴더 경로  
- result_path : Reranker 성능 결과 기록할 경로 

```bash
# 커스텀 설정으로 학습 (예시)
python train_reranker.py \
    --reranker_model_name SFR \
    --retrieval_results_path ../outputs/cypherbench/retrieval_results/SFR \
    --model_path ./models/my_reranker \ 
    --epochs 15 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --result_path ./results/my_reranker
```

### 3. Re-ranker 모델 추론 (Custom)

#### 인자 설명 
- reranker_model_dir : 학습한 Reranker 가 저장되어 있는 경로
- retrieval_results_folder : 이전 단계인 Retriever 을 통해 생성한 (NLQ, Top-20 Candidates) 폴더 경로  
- reranking_results_path : Reranker 을 통해 재정렬한 결과를 저장한 폴더 

```bash
# 커스텀 설정으로 추론 (예시)
python run_reranker.py \
    --reranker_model_dir models/SFR \
    --retrieval_results_folder ../outputs/cypherbench/retrieval_results/SFR \
    --reranking_results_folder ../outputs/cypherbench/reranking_results/SFR
```



## ⚙️ 파라미터 설명

### train_reranker.py 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--reranker_model_name` | `SFR` | Re-ranker 모델 이름 |
| `--retrieval_results_path` | `../outputs/cypherbench/retrieval_results/SFR` | Retriever 결과 폴더 경로 |
| `--model_path` | `./models/SFR` | 모델 저장 경로 |
| `--epochs` | `21` | 학습 에포크 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--learning_rate` | `2e-5` | 학습률 |
| `--result_path` | `./results/SFR` | 결과 저장 경로 |

### run_reranker.py 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--reranker_model_dir` | `models/SFR` | 학습된 Re-ranker 모델 경로 |
| `--retrieval_results_folder` | `../outputs/cypherbench/retrieval_results/SFR` | Retriever 결과 폴더 |
| `--reranking_results_folder` | `../outputs/cypherbench/reranking_results/SFR` | Re-ranker 결과 폴더 |


### 입력 데이터 형식 (Retriever 결과)

Retriever가 생성한 JSON 파일 형식:

```json
{
    "nl_question": "질문 텍스트",
    "candidates": ["candidate1", "candidate2", ..., "candidate20"],
    "label": 5,
    "graph": "graph_name",
    "gold_cypher": "original_cypher_query"
}
```

### 출력 데이터

- **모델 파일**: `models/{reranker_model_name}/`
- **모델 성능 결과**: `results/{reranker_model_name}/reranker_results.csv`
- **Re-ranker 결과**: `../outputs/cypherbench/reranking_results/{model_name}/train.json`, `test.json`

## 📈 평가 메트릭

- **Recall@1, Recall@5, Recall@10, Recall@20**: Top-K 정확도
- **Cross-Encoder Loss**: Cross-Encoder 학습 손실



## 🔄 현재까지 워크플로우

1. **Retriever 학습**: `2.Retrieval/train_retriever.py`로 Retriever 모델 학습
2. **Retriever 추론**: `2.Retrieval/run_retriever.py`로 Top-20 후보 생성
3. **Re-ranker 학습**: `3.Re-ranking/train_reranker.py`로 Cross-Encoder 학습
4. **Re-ranker 추론**: `3.Re-ranking/run_reranker.py`로 최종 재정렬
