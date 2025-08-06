#!/bin/bash

# T5 Slot-Filling 모델 학습 스크립트
# 사용법: ./train.sh [base_model] [train_file] [eval_file] [output_dir] [num_epochs] [batch_size] [learning_rate]

# GPU 설정 (필요시)
export CUDA_VISIBLE_DEVICES=0,1,2,3


# 기본값 설정
BASE_MODEL=${1:-"t5-base"}
TRAIN_FILE=${2:-"./dataset_for_slot_filling/t5_train_data.jsonl"}
EVAL_FILE=${3:-"./dataset_for_slot_filling/t5_test_data.jsonl"}
OUTPUT_DIR=${4:-"./models/${BASE_MODEL}"}
NUM_EPOCHS=${5:-10}
BATCH_SIZE=${6:-8}
LEARNING_RATE=${7:-5e-5}

echo "=== T5 Slot-Filling 모델 학습 시작 ==="
echo "기본 모델: $BASE_MODEL"
echo "훈련 파일: $TRAIN_FILE"
echo "평가 파일: $EVAL_FILE"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "에포크 수: $NUM_EPOCHS"
echo "배치 크기: $BATCH_SIZE"
echo "학습률: $LEARNING_RATE"
echo "================================"

# 출력 디렉토리 생성
mkdir -p $OUTPUT_DIR


# 학습 실행
python train_t5_slot_filler.py \
    --train \
    --base_model $BASE_MODEL \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --gradient_accumulation_steps 4

echo "=== 학습 완료 ==="
echo "모델이 다음 경로에 저장되었습니다: $OUTPUT_DIR" 