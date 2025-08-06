#!/bin/bash

# T5 Slot-Filling 데이터 전처리 스크립트
# 사용법: ./preprocess.sh [input_file] [output_file]

# 기본값 설정
TRAIN_INPUT_FILE=${1:-"../4.Slot-filling-BERT/dataset_for_slot_filling/train_with_literals.jsonl"}
TRAIN_OUTPUT_FILE=${2:-"./dataset_for_slot_filling/t5_train_data.jsonl"}

TEST_INPUT_FILE=${3:-"../4.Slot-filling-BERT/dataset_for_slot_filling/test_with_literals.jsonl"}
TEST_OUTPUT_FILE=${4:-"./dataset_for_slot_filling/t5_test_data.jsonl"}

echo "=== T5 Slot-Filling 데이터 전처리 시작 ==="
echo "학습 데이터 입력 파일: $TRAIN_INPUT_FILE"
echo "평가 데이터 입력 파일: $TEST_INPUT_FILE"
echo "================================"

# 출력 디렉토리 생성
mkdir -p $(dirname $TRAIN_OUTPUT_FILE)
mkdir -p $(dirname $TEST_OUTPUT_FILE)

# 데이터 전처리 실행
python train_t5_slot_filler.py \
    --preprocess \
    --input_file $TRAIN_INPUT_FILE \
    --preprocessed_output $TRAIN_OUTPUT_FILE

python train_t5_slot_filler.py \
    --preprocess \
    --input_file $TEST_INPUT_FILE \
    --preprocessed_output $TEST_OUTPUT_FILE

echo "=== 데이터 전처리 완료 ==="
echo "전처리된 데이터가 다음 파일에 저장되었습니다: $TRAIN_OUTPUT_FILE"
echo "전처리된 데이터가 다음 파일에 저장되었습니다: $TEST_OUTPUT_FILE"
