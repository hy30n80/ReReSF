#!/bin/bash

# Retriever 모델 추론 스크립트
# 사용법: ./run.sh [model_name] [epoch] [train_file_path]

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 기본값 설정
MODEL_NAME=${1:-"SFR"}
EPOCH=${2:-21}
DATASET_PATH=${3:-"../datasets/cypherbench"}
BATCH_SIZE=${4:-128}
MODEL_PATH=${5:-"./models/SFR"}


echo "=== Retriever 모델 추론 시작 ==="
echo "타겟 데이터셋 경로: $DATASET_PATH"
echo "배치 크기: $BATCH_SIZE"
echo "모델 경로: $MODEL_PATH"
echo "================================"

# 결과 디렉토리 생성
mkdir -p results/$MODEL_NAME
mkdir -p ../outputs/cypherbench/retrieval_results/$MODEL_NAME

# 추론 실행
python run_retriever.py \
    --dataset_path $DATASET_PATH \
    --result_path results/$MODEL_NAME \
    --output_path ../outputs/cypherbench/retrieval_results/$MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --model_path $MODEL_PATH

echo "=== 추론 완료 ==="
echo "Cross-Encoder 데이터가 ../outputs/cypherbench/retrieval_results/$MODEL_NAME/ 에 생성되었습니다."
echo "평가 결과가 results/$MODEL_NAME/ 에 저장되었습니다." 