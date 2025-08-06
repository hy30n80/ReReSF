#!/bin/bash

# Re-ranker 추론 스크립트
# 사용법: ./run.sh [reranker_model_dir] [input_folder] [output_folder]

export CUDA_VISIBLE_DEVICES=0,1,2,3


# 기본값 설정
RERANKER_MODEL_DIR=${1:-"models/SFR"}
INPUT_FOLDER=${2:-"../outputs/cypherbench/retrieval_results/SFR"}
OUTPUT_FOLDER=${3:-"../outputs/cypherbench/reranking_results/SFR"}

echo "=== Re-ranker 추론 시작 ==="
echo "Reranker 모델: $RERANKER_MODEL_DIR"
echo "입력 폴더: $INPUT_FOLDER"
echo "출력 폴더: $OUTPUT_FOLDER"
echo "================================"

# 출력 폴더 생성
mkdir -p $OUTPUT_FOLDER

# 추론 실행
python run_reranker.py \
    --reranker_model_dir $RERANKER_MODEL_DIR \
    --retrieval_results_folder $INPUT_FOLDER \
    --reranker_results_folder $OUTPUT_FOLDER

echo "=== 추론 완료 ==="
echo "결과가 $OUTPUT_FOLDER/train.json, $OUTPUT_FOLDER/test.json 에 저장되었습니다." 