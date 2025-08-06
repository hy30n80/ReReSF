#!/bin/bash

# Slot-Filling 추론 스크립트
# 사용법: ./run.sh [model_path] [input_file] [num_examples]

# 기본값 설정
MODEL_PATH=${1:-"./models/bert-base-cased"}
INPUT_FILE=${2:-"./dataset_for_slot_filling/test_with_literals.jsonl"}
NUM_EXAMPLES=${3:-"all"}
OUTPUT_TXT_FILE=${4:-"./results/inference_results.txt"}
OUTPUT_JSONL_FILE=${5:-"./results/inference_results.jsonl"}

echo "=== Slot-Filling 추론 시작 ==="
echo "모델 경로: $MODEL_PATH"
echo "입력 파일: $INPUT_FILE"
echo "추론 예시 수: $NUM_EXAMPLES"
echo "출력 TXT 파일: $OUTPUT_TXT_FILE"
echo "출력 JSONL 파일: $OUTPUT_JSONL_FILE"
echo "================================"

# 출력 디렉토리 생성
mkdir -p $(dirname $OUTPUT_TXT_FILE)
mkdir -p $(dirname $OUTPUT_JSONL_FILE)

# GPU 설정 (필요시)
export CUDA_VISIBLE_DEVICES=0

# 추론 실행
if [ "$NUM_EXAMPLES" = "all" ]; then
    python run_slot-filler.py \
        --model_path $MODEL_PATH \
        --input_file $INPUT_FILE \
        --output_txt_file $OUTPUT_TXT_FILE \
        --output_jsonl_file $OUTPUT_JSONL_FILE
else
    python run_slot-filler.py \
        --model_path $MODEL_PATH \
        --input_file $INPUT_FILE \
        --output_txt_file $OUTPUT_TXT_FILE \
        --output_jsonl_file $OUTPUT_JSONL_FILE \
        --num_examples $NUM_EXAMPLES
fi

echo "=== 추론 완료 ==="
echo "결과가 다음 파일에 저장되었습니다:"
echo "  - 상세 로그: $OUTPUT_TXT_FILE"
echo "  - 기계 읽기 가능: $OUTPUT_JSONL_FILE"


