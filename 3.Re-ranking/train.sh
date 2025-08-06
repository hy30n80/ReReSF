#!/bin/bash

# GPU 설정 (메모리 부족 문제 해결을 위해 GPU 수 줄임)
export CUDA_VISIBLE_DEVICES=0,1,2,3


# 기본값 설정
EPOCHS=${1:-21}
BATCH_SIZE=${2:-16}
LEARNING_RATE=${3:-2e-5}
MODEL_NAME=${4:-"SFR"}
RETRIEVAL_RESULTS_PATH=${5:-"../outputs/cypherbench/retrieval_results/SFR"}

echo "=== Re-ranker 모델 학습 시작 ==="
echo "에포크: $EPOCHS"
echo "배치 크기: $BATCH_SIZE"
echo "학습률: $LEARNING_RATE"
echo "Reranker 모델: $MODEL_NAME"
echo "Retrieval 결과 파일 (=Reranker 학습 데이터) 경로 : $RETRIEVAL_RESULTS_PATH"
echo "================================"

# 모델 저장 디렉토리 생성
mkdir -p models/$MODEL_NAME
mkdir -p results/$MODEL_NAME

# 학습 실행
python train_reranker.py \
    --retrieval_results_path $RETRIEVAL_RESULTS_PATH \
    --model_path models/$MODEL_NAME \
    --reranker_model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --result_path results/$MODEL_NAME

echo "=== 학습 완료 ==="
echo "모델이 models/$MODEL_NAME/ 에 저장되었습니다."
echo "결과가 results/${MODEL_NAME}_seen.csv, results/${MODEL_NAME}_unseen.csv 에 저장되었습니다." 