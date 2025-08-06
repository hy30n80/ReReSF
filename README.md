# ReReSF: Retrieval-Re-ranking-Slot-Filling 시스템

ReReSF는 Text-to-Cypher 검색을 위한 3단계 파이프라인 시스템입니다. Dense Passage Retrieval, Cross-Encoder Re-ranking, 그리고 Slot-Filling을 순차적으로 수행하여 자연어 질문을 Cypher 쿼리로 변환합니다.

## 📁 시스템 구조

```
ReReSF/
├── 2.Retrieval/                    # 1단계: Dense Passage Retrieval
│   ├── train_retriever.py         # Retriever 모델 학습
│   ├── run_retriever.py           # Retriever 모델 추론
│   ├── train.sh                   # 학습 실행 스크립트
│   ├── run.sh                     # 추론 실행 스크립트
│   ├── eval.py                    # 평가 함수
│   ├── models/                    # 학습된 모델 저장
│   ├── results/                   # 추론 결과 저장
│   └── README.md                  # Retriever 모듈 문서
├── 3.Re-ranking/                  # 2단계: Cross-Encoder Re-ranking
│   ├── train_reranker.py          # Re-ranker 모델 학습
│   ├── run_reranker.py            # Re-ranker 모델 추론
│   ├── train.sh                   # 학습 실행 스크립트
│   ├── run.sh                     # 추론 실행 스크립트
│   ├── models/                    # 학습된 모델 저장
│   ├── results/                   # 추론 결과 저장
│   └── README.md                  # Re-ranker 모듈 문서
├── 4.Slot-filling-BERT/           # 3단계: BERT 기반 Slot-Filling
│   ├── train_slot-filler.py       # BERT Slot-Filling 모델 학습
│   ├── run_slot-filler.py         # BERT Slot-Filling 모델 추론
│   ├── extract_literal.py         # LITERAL 정보 추출
│   ├── process_literals.sh        # LITERAL 데이터 구축 스크립트
│   ├── train.sh                   # 학습 실행 스크립트
│   ├── run.sh                     # 추론 실행 스크립트
│   ├── models/                    # 학습된 모델 저장
│   ├── results/                   # 추론 결과 저장
│   ├── dataset_for_slot_filling/  # Slot-Filling 데이터셋
│   └── README.md                  # BERT Slot-Filling 모듈 문서
├── 4.Slot-filling-T5/             # 3단계: T5 기반 Slot-Filling
│   ├── train_t5_slot_filler.py    # T5 Slot-Filling 모델 학습
│   ├── run_t5_slot_filler.py      # T5 Slot-Filling 모델 추론
│   ├── preprocess.sh              # T5 학습용 데이터 전처리
│   ├── train.sh                   # 학습 실행 스크립트
│   ├── run.sh                     # 추론 실행 스크립트
│   ├── models/                    # 학습된 모델 저장
│   ├── results/                   # 추론 결과 저장
│   └── README.md                  # T5 Slot-Filling 모듈 문서
├── datasets/                       # 원본 데이터셋
├── outputs/                        # 중간 결과 및 최종 출력
└── README.md                      # 이 파일 (통합 문서)
```

## 🚀 빠른 시작

### 전체 워크플로우 (BERT 기반)

```bash
# 1단계: Retriever 학습 및 추론
cd 2.Retrieval
./train.sh
./run.sh

# 2단계: Re-ranker 학습 및 추론
cd ../3.Re-ranking
./train.sh
./run.sh

# 3단계: BERT Slot-Filling 학습 및 추론
cd ../4.Slot-filling-BERT
./process_literals.sh
./train.sh
./run.sh
```

### 전체 워크플로우 (T5 기반)

```bash
# 1단계: Retriever 학습 및 추론
cd 2.Retrieval
./train.sh
./run.sh

# 2단계: Re-ranker 학습 및 추론
cd ../3.Re-ranking
./train.sh
./run.sh

# 3단계: T5 Slot-Filling 학습 및 추론
cd ../4.Slot-filling-T5
./preprocess.sh
./train.sh
./run.sh
```

## 📊 시스템 개요

### 1단계: Retrieval (Dense Passage Retrieval)
- **모델**: SFR-Embedding-Code-400M_R
- **목적**: 자연어 질문에 대해 관련된 Top-20 Cypher 쿼리 후보 검색
- **입력**: 자연어 질문
- **출력**: Top-20 후보 Cypher 쿼리들

### 2단계: Re-ranking (Cross-Encoder)
- **모델**: SFR-Embedding-Code-400M_R (Cross-Encoder)
- **목적**: Top-20 후보들을 재정렬하여 최적의 Top-1 선택
- **입력**: 자연어 질문 + Top-20 Cypher 후보들
- **출력**: 재정렬된 Top-1 Cypher 쿼리

### 3단계: Slot-Filling
- **BERT 버전**: Conditional Span Prediction
- **T5 버전**: Text-to-Text Generation
- **목적**: 마스킹된 Cypher 쿼리의 LITERAL 값들을 추출
- **입력**: 마스킹된 Cypher 쿼리 + 자연어 질문
- **출력**: 완성된 Cypher 쿼리

## 🔧 환경 설정

### 방법 1: Conda 환경 재현 (권장)

현재 사용 중인 `text2cypher` 가상환경을 그대로 재현할 수 있습니다:

```bash
# 1. Anaconda/Miniconda 설치 (이미 설치되어 있다면 생략)
# https://docs.conda.io/en/latest/miniconda.html

# 2. 환경 재현
cd ReReSF
conda env create -f environment.yml

# 3. 환경 활성화
conda activate text2cypher
```

### 방법 2: 수동 설치

필수 패키지만 설치하려면:

```bash
# 1. Python 가상환경 생성
python -m venv text2cypher
source text2cypher/bin/activate  # Linux/Mac
# 또는
text2cypher\Scripts\activate     # Windows

# 2. 필수 패키지 설치
pip install -r requirements.txt
# 또는 개별 설치:
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
# pip install transformers==4.53.2 sentence-transformers==5.0.0
# pip install numpy==1.26.4 tqdm==4.67.1 tabulate==0.9.0
# pip install tensorboard==2.19.0 psutil==5.9.0

### 주요 패키지 버전

| 패키지 | 버전 | 설치 방법 |
|--------|------|----------|
| PyTorch | 2.6.0 | conda/pip |
| Transformers | 4.53.2 | pip |
| Sentence-Transformers | 5.0.0 | pip |
| NumPy | 1.26.4 | conda |
| tqdm | 4.67.1 | conda |
| tabulate | 0.9.0 | pip |
| TensorBoard | 2.19.0 | conda |
| psutil | 5.9.0 | conda |



## 📈 평가 메트릭

### Retrieval 단계
- **Recall@K**: Top-K 후보 중 정답 포함 비율
- **MRR (Mean Reciprocal Rank)**: 정답의 순위 역수 평균

### Re-ranking 단계
- **Top-1 정확도**: 최종 선택된 쿼리의 정확도
- **MRR**: 재정렬 후 정답의 순위 역수 평균

### Slot-Filling 단계
- **Exact Match Ratio (EMR)**: 전체 정확도
- **Slot-Filling 정확도 (EM)**: LITERAL 값 정확도
- **실행 쿼리 정확도 (EA)**: 재구성된 Cypher 쿼리 정확도
- **LITERAL_C별 EMR**: 각 LITERAL 개수별 정확도

## 💡 사용 예시

### 실험 1: 기본 워크플로우 (BERT)

```bash
# 전체 파이프라인 실행
cd ReReSF

# 1단계: Retrieval
cd 2.Retrieval
./train.sh
./run.sh

# 2단계: Re-ranking
cd ../3.Re-ranking
./train.sh
./run.sh

# 3단계: BERT Slot-Filling
cd ../4.Slot-filling-BERT
./process_literals.sh
./train.sh
./run.sh
```


## 📋 데이터 형식

### 입력 데이터 (CypherBench 형식)

```json
{
    "qid": "5ccda891-9f3c-4787-993c-0c72e07de25c",
    "graph": "geography",
    "gold_cypher": "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: 'Natara'}) WITH DISTINCT n WHERE n.area_km2 < 390000 RETURN n.name",
    "nl_question": "What are the names of lakes situated in countries through which the Natara River flows, with an area of less than 390,000 square kilometers?",
    "from_template": {
        "match_category": "basic_(n)-(m0)-(m1*)",
        "match_cypher": "MATCH (n)-[r0]->(m0)<-[r1]-(m1<name>)",
        "return_pattern_id": "n_where",
        "return_cypher": "${match_cypher} WITH DISTINCT n WHERE ${condition} RETURN n.name"
    },
    "masked_question": "What are the names of lakes situated in countries through which the [LITERAL] River flows, with an area of less than 390,000 square kilometers?",
    "masked_cypher": "MATCH (n:Lake)-[r0:locatedIn]->(m0:Country)<-[r1:flowsThrough]-(m1:River {name: '[LITERAL]'}) WITH DISTINCT n WHERE n.area_km2 < [LITERAL] RETURN n.name",
    "negative_cypher": [],
    "hard_negative_cypher": [...]
}
```

### 중간 결과 (Retrieval)

```json
{
    "qid": "ac09e4a1-a550-433e-9887-ceda1bdf235f",
    "graph": "geography",
    "gold_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: 'India'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "nl_question": "What are the names of mountain ranges that include mountains located in India, and how many such mountains are part of each range?",
    "masked_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: '[LITERAL]'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "candidates": [
        "MATCH (n:Mountain)-[r0:partOf]->(m0:MountainRange) WITH DISTINCT n RETURN COUNT(DISTINCT n.name)",
        "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain) WITH DISTINCT n RETURN COUNT(DISTINCT n.name)",
        "MATCH (n:Mountain)-[r0:partOf]->(m0:MountainRange) WITH DISTINCT n RETURN n.name, n.elevation_m",
        ...
    ],
    "label": 11
}
```

### 중간 결과 (Re-ranking)

```json
{
    "qid": "ac09e4a1-a550-433e-9887-ceda1bdf235f",
    "graph": "geography",
    "gold_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: 'India'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "nl_question": "What are the names of mountain ranges that include mountains located in India, and how many such mountains are part of each range?",
    "masked_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: '[LITERAL]'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "candidates": [
        "MATCH (n:Mountain)-[r0:partOf]->(m0:MountainRange) WITH DISTINCT n RETURN COUNT(DISTINCT n.elevation_m)",
        "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain) WITH DISTINCT n RETURN COUNT(DISTINCT n.name)",
        "MATCH (n:Mountain)-[r0:partOf]->(m0:MountainRange) WITH DISTINCT n RETURN COUNT(DISTINCT n)",
        ...
    ],
    "label": 14
}
```

### 최종 결과 (Slot-Filling)

```json
{
    "nl_question": "What are the names of mountain ranges that include mountains located in India, and how many such mountains are part of each range?",
    "masked_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: '[LITERAL]'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "predicted_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: 'India'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "gold_cypher": "MATCH (n:MountainRange)<-[r0:partOf]-(m0:Mountain)-[r1:locatedIn]->(m1:Country {name: 'India'}) WITH n, count(DISTINCT m0) AS num RETURN n.name, num",
    "is_correct": true,
    "literals": {
        "India": "India"
    }
}
```

## 🔍 모듈별 상세 정보

### 2.Retrieval
- **모델**: SFR-Embedding-Code-400M_R
- **방식**: Dense Passage Retrieval
- **입력**: 자연어 질문
- **출력**: Top-20 Cypher 후보들
- **평가**: Recall@K, MRR

### 3.Re-ranking
- **모델**: SFR-Embedding-Code-400M_R (Cross-Encoder)
- **방식**: Cross-Encoder Re-ranking
- **입력**: 자연어 질문 + Top-20 Cypher 후보들
- **출력**: 재정렬된 Top-1 Cypher 쿼리
- **평가**: Top-1 정확도, MRR

### 4.Slot-filling-BERT
- **모델**: BERT-base-cased
- **방식**: Conditional Span Prediction
- **입력**: 마스킹된 Cypher + 자연어 질문
- **출력**: LITERAL 값들의 span 위치
- **평가**: EM, EA, EMR

### 4.Slot-filling-T5
- **모델**: T5-base/large
- **방식**: Text-to-Text Generation
- **입력**: 자연어 질문 (T5 형식)
- **출력**: LITERAL 값들의 텍스트
- **평가**: EM, EA, EMR, Latency, Throughput


## 🔄 전체 워크플로우 요약

1. **Retrieval**: 자연어 질문 → Top-20 Cypher 후보 검색
2. **Re-ranking**: Top-20 후보 → Top-1 최적 후보 선택
3. **Slot-Filling**: Top-1 마스킹된 쿼리 → 완성된 Cypher 쿼리

이 3단계 파이프라인을 통해 자연어 질문을 정확한 Cypher 쿼리로 변환할 수 있습니다.

