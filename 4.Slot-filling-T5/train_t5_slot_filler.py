import json
import argparse
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm
import re

# --- 1. 설정 및 데이터셋 클래스 정의 ---

class CFG:
    MAX_LENGTH = 512
    LITERAL_PLACEHOLDER = "[LITERAL]"

class T5SlotFillingDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"데이터 로딩 중: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        print(f"로딩된 데이터 수: {len(self.data)}개")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_text = item['target']
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 타겟 토크나이징
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# --- 2. T5 모델 학습 함수 ---
def train_t5_slot_filling(args):
    print("T5 기반 Slot Filling 모델 학습을 시작합니다...")
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    
    # 토크나이저 및 모델 로딩
    print(f"T5 토크나이저 로딩: {args.base_model}")
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    
    # 특수 토큰 추가
    if CFG.LITERAL_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([CFG.LITERAL_PLACEHOLDER])
        print(f"특수 토큰 추가: {CFG.LITERAL_PLACEHOLDER}")
    
    print(f"T5 모델 로딩: {args.base_model}")
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    
    # 토크나이저 크기에 맞춰 모델 임베딩 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # 데이터셋 생성
    print("데이터셋 생성 중...")
    train_dataset = T5SlotFillingDataset(args.train_file, tokenizer, CFG.MAX_LENGTH)
    
    if args.eval_file:
        eval_dataset = T5SlotFillingDataset(args.eval_file, tokenizer, CFG.MAX_LENGTH)
    else:
        eval_dataset = None
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        report_to=None,  # wandb 비활성화
    )
    
    # Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"모델 저장 중: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("학습 완료!")

# --- 3. 데이터 전처리 함수 ---
def preprocess_data_for_t5(input_file, output_file):
    """기존 데이터를 T5 학습용으로 전처리"""
    print(f"T5 학습용 데이터 전처리 중: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    t5_data = []
    for item in data:
        # if item.get('label') == 0:  # 유효한 데이터만
        masked_cypher = item["masked_cypher"]
        nl_question = item["nl_question"]
        ground_truth_answers = [ans['text'] for ans in item['answers']]
        
        # 더 간단한 T5 입력/출력 형식으로 변환
        input_text = f"Question: {nl_question}"
        target_text = " ".join([f"Answer{i+1}: {ans}" for i, ans in enumerate(ground_truth_answers)])
        
        t5_data.append({
            "input": input_text,
            "target": target_text
        })
    
    # T5 학습 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in t5_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"T5 학습 데이터 전처리 완료: {len(t5_data)}개 샘플")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 데이터 전처리 모드
    parser.add_argument("--preprocess", action="store_true", help="Preprocess data for T5 training")
    parser.add_argument("--input_file", type=str, help="Input file for preprocessing")
    parser.add_argument("--preprocessed_output", type=str, help="Output file for preprocessed data")
    
    # 학습 모드
    parser.add_argument("--train", action="store_true", help="Train T5 model")
    parser.add_argument("--base_model", type=str, default="t5-base", help="Base T5 model to fine-tune")
    parser.add_argument("--train_file", type=str, help="Training data file")
    parser.add_argument("--eval_file", type=str, help="Evaluation data file (optional)")
    parser.add_argument("--output_dir", type=str, help="Output directory for trained model")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    if args.preprocess:
        if not args.input_file or not args.preprocessed_output:
            print("Error: --input_file and --preprocessed_output are required for preprocessing")
            exit(1)
        preprocess_data_for_t5(args.input_file, args.preprocessed_output)
    
    elif args.train:
        if not args.train_file or not args.output_dir:
            print("Error: --train_file and --output_dir are required for training")
            exit(1)
        train_t5_slot_filling(args)
    
    else:
        print("Error: Please specify either --preprocess or --train")
        exit(1)

# 사용 예시:
# 1. 데이터 전처리:
# python train_t5_slot_filling.py \
#     --preprocess \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/train_with_literals.jsonl \
#     --preprocessed_output ./t5_train_data.jsonl

# python train_t5_slot_filling.py \
#     --preprocess \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --preprocessed_output ./t5_eval_data.jsonl

# 2. 모델 학습:
# CUDA_VISIBLE_DEVICES=0 python train_t5_slot_filling.py \
#     --train \
#     --base_model t5-large \
#     --train_file ./t5_train_data.jsonl \
#     --eval_file ./t5_eval_data.jsonl \
#     --output_dir ./t5_slot_filling_model/t5-large \
#     --num_epochs 3 \
#     --batch_size 8 \
#     --learning_rate 5e-5 