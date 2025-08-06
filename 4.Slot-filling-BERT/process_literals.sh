#!/bin/bash

# Slot-Filling 데이터 구축 스크립트
# Train Set과 Test Set에 대해 LITERAL 정보를 추출하는 스크립트

# 기본값 설정
OUTPUT_BASE_PATH=${2:-"./dataset_for_slot_filling"}
TRAIN_SET_PATH=${3:-"../datasets/cypherbench/train.json"}
RERANKING_RESULTS_PATH=${3:-"../outputs/cypherbench/reranking_results/SFR"}

echo "=== Slot-Filling 데이터 구축 시작 ==="
echo "원본 Train Set 경로: $TRAIN_SET_PATH"
echo "Retrieval & Reranking 거쳐서 살아남은 Test Set 경로: $RERANKING_RESULTS_PATH"
echo "Slot Filling 을 위한 데이터 셋 저장 경로: $OUTPUT_BASE_PATH"
echo "================================"

# 출력 디렉토리 생성
mkdir -p $OUTPUT_BASE_PATH

# 모든 Train set 에 대해서 Slot-Filling 학습 데이터 구축 처리
echo "📝 Train set 처리 중..."
if [ -f "$TRAIN_SET_PATH" ]; then
    python extract_literals.py \
        --input_file "$TRAIN_SET_PATH" \
        --output_file "$OUTPUT_BASE_PATH/train_with_literals.jsonl"
    echo "✅ Train set 처리 완료: $OUTPUT_BASE_PATH/train_with_literals.jsonl"
else
    echo "⚠️  Warning: ../datasets/cypherbench/train.json 파일을 찾을 수 없습니다."
fi
    
# Retrieval & Reranking 거쳐서 살아남은 Test Set 에 대해서 Slot-Filling 추론 데이터 구축 처리
echo "📝 Test set 처리 중..."
if [ -f "$RERANKING_RESULTS_PATH/test.json" ]; then
    python extract_literals.py \
        --input_file "$RERANKING_RESULTS_PATH/test.json" \
        --output_file "$OUTPUT_BASE_PATH/test_with_literals.jsonl"
    echo "✅ Test set 처리 완료: $OUTPUT_BASE_PATH/test_with_literals.jsonl"
else
    echo "⚠️  Warning: $RERANKING_RESULTS_PATH/test.json 파일을 찾을 수 없습니다."
fi

echo "=== Slot-Filling 데이터 구축 완료 ==="
echo "결과 파일들:"
echo "  - Train: $OUTPUT_BASE_PATH/train_with_literals.jsonl"
echo "  - Test: $OUTPUT_BASE_PATH/test_with_literals.jsonl" 