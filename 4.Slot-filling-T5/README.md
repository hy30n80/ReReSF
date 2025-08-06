# T5 Slot-Filling 학습 및 추론 시스템

이 시스템은 Text-to-Cypher 검색을 위한 T5 기반 Slot-Filling 모델을 학습하고 추론하는 코드입니다. LITERAL 정보를 추출하여 마스킹된 Cypher 쿼리를 완성하는 작업을 수행합니다.

## 📁 시스템 구조

```
4.Slot-filling-T5/
├── train_t5_slot_filler.py    # T5 Slot-Filling 모델 학습
├── run_t5_slot_filler.py      # T5 Slot-Filling 모델 추론
├── preprocess.sh               # T5 학습용 데이터 전처리 스크립트
├── train.sh                    # 학습 실행 스크립트
├── run.sh                      # 추론 실행 스크립트
├── models/                     # 학습된 모델 저장 디렉토리
├── results/                    # 추론 결과 저장 디렉토리
├── logs/                       # 학습 로그 저장 디렉토리
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 1. 스크립트 사용 (내가 CypherBench 에서 사용했던 Bash Shell)

```bash
# T5 학습 및 평가용 데이터 전처리 (Train Set)
./preprocess.sh

# T5 Slot-Filling 모델 학습
./train.sh

# T5 Slot-Filling 모델 추론
./run.sh
```

### 2. T5 학습용 데이터 전처리 (Custom)

```bash
# 데이터 전처리 실행
python train_t5_slot_filler.py \
    --preprocess \
    --input_file $TRAIN_INPUT_FILE \
    --preprocessed_output $TRAIN_OUTPUT_FILE

python train_t5_slot_filler.py \
    --preprocess \
    --input_file $TEST_INPUT_FILE \
    --preprocessed_output $TEST_OUTPUT_FILE
```

### 3. T5 Slot-Filling 모델 학습 (Custom)

#### 인자 설명
- train_file : T5 Slot-Filling 학습 데이터 (전처리된 형식)
- eval_file : T5 Slot-Filling 평가할 데이터 (전처리된 형식)

```bash
# 커스텀 설정으로 학습
python train_t5_slot_filler.py \
    --train \
    --base_model t5-large \
    --train_file ./t5_train_data.jsonl \
    --eval_file ./t5_eval_data.jsonl \
    --output_dir ./my_t5_model \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 3e-5
```

### 4. T5 Slot-Filling 모델 추론 (Custom)

#### 인자 설명
- model_path : 학습한 T5 Slot-Filling 모델 경로
- input_file : T5 Slot-Filling 평가할 데이터 (전처리된 형식)

```bash
# 커스텀 설정으로 추론
python run_t5_slot_filler.py \
    --model_path ./my_t5_model \
    --input_file ./t5_eval_data.jsonl \
    --output_txt_file ./my_results/t5_results.txt \
    --output_jsonl_file ./my_results/t5_results.jsonl \
    --batch_size 8 \
    --measure_performance
```


## 📋 데이터 형식

### 입력 데이터 형식 (T5 학습용)

```json
{
    "input": "Question: What is the Habitat name?",
    "target": "Answer1: floodplain"
}
```

### 원본 LITERAL 데이터 형식

```json
{
    "nl_question": "질문 텍스트",
    "masked_cypher": "MATCH (n:Taxon)-[r0:inhabits]->(m0:Habitat {name: '[LITERAL]'})",
    "gold_cypher": "MATCH (n:Taxon)-[r0:inhabits]->(m0:Habitat {name: 'floodplain'})",
    "LITERAL_C": 1,
    "LITERAL_V": ["floodplain"],
    "LITERAL_Q": ["What is the Habitat name?"],
    "answers": [
        {
            "text": "floodplain",
            "start_char": 15
        }
    ]
}
```

### 출력 데이터

- **모델 파일**: `models/t5_slot_filling_model/`
- **학습 로그**: `logs/` 디렉토리
- **추론 결과**: 
  - `results/t5_inference_results.txt` (상세 로그)
  - `results/t5_inference_results.jsonl` (기계 읽기 가능)
  - `results/latency_metrics.csv` (성능 측정 결과)



## 💡 사용 예시

### 실험 1: 기본 워크플로우

```bash
# 1. T5 학습 및 평가용 데이터 전처리
./preprocess.sh

# 2. T5 모델 학습
./train.sh

# 3. T5 모델 추론
./run.sh
```

## 🔄 현재까지 워크플로우
1. **Retriever 학습**: `2.Retrieval/train_retriever.py`로 Retriever 모델 학습
2. **Retriever 추론**: `2.Retrieval/run_retriever.py`로 Top-20 후보 생성
3. **Re-ranker 학습**: `3.Re-ranking/train_reranker.py`로 Cross-Encoder 학습
4. **Re-ranker 추론**: `3.Re-ranking/run_reranker.py`로 최종 재정렬
5. **T5 데이터 전처리**: `4.Slot-filling-T5/preprocess.sh`로 T5 학습용 데이터 생성
6. **T5 Slot-Filling 학습**: `4.Slot-filling-T5/train_t5_slot_filler.py`로 T5 모델 학습
7. **T5 Slot-Filling 추론**: `4.Slot-filling-T5/run_t5_slot_filler.py`로 최종 LITERAL 생성

