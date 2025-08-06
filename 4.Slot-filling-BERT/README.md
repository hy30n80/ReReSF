# Slot-Filling 학습 및 추론 시스템

이 시스템은 Text-to-Cypher 검색을 위한 Slot-Filling 모델을 학습하고 추론하는 코드입니다. LITERAL 정보를 추출하여 마스킹된 Cypher 쿼리를 완성하는 작업을 수행합니다.

## 📁 시스템 구조

```
4.Slot-filling/
├── train_slot-filler.py    # Slot-Filling 모델 학습
├── run_slot-filler.py      # Slot-Filling 모델 추론
├── extract_literal.py      # Slot-Filling 학습 및 평가를 위한 데이터 형태 추출 (LITERAL 정보 추출)
├── process_literals.sh     # LITERAL 데이터 구축 스크립트
├── train.sh                # 학습 실행 스크립트
├── run.sh                  # 추론 실행 스크립트
├── models/                 # 학습된 모델 저장 디렉토리
├── results/                # 추론 결과 저장 디렉토리
├── logs/                   # 학습 로그 저장 디렉토리
├── dataset_for_slot_filling/ # Slot-Filling 데이터셋
└── README.md              # 이 파일
```

## 🚀 빠른 시작

### 1. 스크립트 사용 (내가 CypherBench 에서 사용했던 Bash Shell)

```bash
# LITERAL 데이터 구축
./process_literals.sh

# Slot-Filling 모델 학습
./train.sh

# Slot-Filling 모델 추론
./run.sh
```

### 2. LITERAL 데이터 구축 (Custom)

```bash
# 커스텀 설정으로 LITERAL 데이터 구축 (Slot Filling 을 학습 및 평가하기 적합한 구조 추출)
python extract_literal.py \
    --input_file ../outputs/cypherbench/reranking_results/SFR/test.json \ 
    --output_file ./dataset_for_slot_filling/test_with_literals.jsonl
```

### 3. Slot-Filling 모델 학습 (Custom)

#### 인자 설명
- train_file : Slot-Filling 학습 데이터
- eval_file : Slot-Filling 평가할 데이터 (Reranking 과정까지 거쳐서 Top-1 Candidate 에 GT 가 있는 애들)

```bash
# 커스텀 설정으로 학습
python train_slot-filler.py \
    --train_file train_with_literals.jsonl \
    --eval_file test_with_literals.jsonl \
    --model_name bert-base-cased \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --batch_size 16 \
    --max_length 512 \
    --output_dir ./my_results \
    --logging_dir ./my_logs \
    --save_model_dir ./my_model \
    --num_test_examples 5000
```

### 4. Slot-Filling 모델 추론 (Custom)

#### 인자 설명
- model_path : 학습한 Slot-Filling 모델 경로
- input_file : Slot-Filling 평가할 데이터 (Reranking 과정까지 거쳐서 Top-1 Candidate 에 GT 가 있는 애들)

```bash
# 커스텀 설정으로 추론
python run_slot-filler.py \
    --model_path ./my_model \
    --input_file ./my_dataset/test_with_literals.jsonl \
    --output_txt_file ./my_results/results.txt \
    --output_jsonl_file ./my_results/results.jsonl \
    --num_examples 100
```


## 📋 데이터 형식

### 입력 데이터 형식 (LITERAL 정보 포함)

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

- **모델 파일**: `models/{model_name}/`
- **학습 로그**: `logs/` 디렉토리
- **추론 결과**: 
  - `results/inference_results.txt` (상세 로그)
  - `results/inference_results.jsonl` (기계 읽기 가능)


## 🔄 현재까지 워크플로우

1. **Retriever 학습**: `2.Retrieval/train_retriever.py`로 Retriever 모델 학습
2. **Retriever 추론**: `2.Retrieval/run_retriever.py`로 Top-20 후보 생성
3. **Re-ranker 학습**: `3.Re-ranking/train_reranker.py`로 Cross-Encoder 학습
4. **Re-ranker 추론**: `3.Re-ranking/run_reranker.py`로 최종 재정렬
5. **LITERAL 데이터 구축**: `4.Slot-filling/process_literals.sh`로 LITERAL 정보 추출
6. **Slot-Filling 학습**: `4.Slot-filling/train_slot-filler.py`로 Slot-Filling 모델 학습
7. **Slot-Filling 추론**: `4.Slot-filling/run_slot-filler.py`로 최종 LITERAL 추출
