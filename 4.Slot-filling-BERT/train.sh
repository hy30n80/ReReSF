#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Slot-Filling 모델 학습 스크립트
# 사용법: ./train.sh [train_file] [eval_file] [model_name] [learning_rate] [num_epochs] [batch_size]

# 기본값 설정
TRAIN_FILE=${1:-"./dataset_for_slot_filling/train_with_literals.jsonl"}
EVAL_FILE=${2:-"./dataset_for_slot_filling/test_with_literals.jsonl"}
MODEL_NAME=${3:-"bert-base-cased"}
LEARNING_RATE=${4:-2e-5}
NUM_EPOCHS=${5:-10}
BATCH_SIZE=${6:-8}
MAX_LENGTH=${7:-512}
OUTPUT_DIR=${8:-"./results"}
LOGGING_DIR=${9:-"./logs"}  
SAVE_MODEL_DIR=${10:-"./models/${MODEL_NAME}"}
NUM_TEST_EXAMPLES=${11:-10000}

echo "=== Slot-Filling 모델 학습 시작 ==="
echo "훈련 데이터: $TRAIN_FILE"
echo "평가 데이터: $EVAL_FILE"
echo "모델: $MODEL_NAME"
echo "학습률: $LEARNING_RATE"
echo "에포크: $NUM_EPOCHS"
echo "배치 크기: $BATCH_SIZE"
echo "최대 길이: $MAX_LENGTH"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 디렉토리: $LOGGING_DIR"
echo "모델 저장 디렉토리: $SAVE_MODEL_DIR"
echo "테스트 예시 수: $NUM_TEST_EXAMPLES"
echo "================================"

# 디렉토리 생성
mkdir -p $OUTPUT_DIR
mkdir -p $LOGGING_DIR
mkdir -p $SAVE_MODEL_DIR

# 학습 실행
python train_slot-filler.py \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --model_name $MODEL_NAME \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --save_model_dir $SAVE_MODEL_DIR \
    --num_test_examples $NUM_TEST_EXAMPLES

echo "=== 학습 완료 ==="
echo "모델이 $SAVE_MODEL_DIR 에 저장되었습니다."
echo "추론 결과가 inference_results.txt 에 저장되었습니다." 