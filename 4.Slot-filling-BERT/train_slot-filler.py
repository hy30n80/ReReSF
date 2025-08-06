import torch
import torch.nn as nn
import datasets
from datasets import Dataset
import json
import os
import argparse
from transformers import (
    AutoTokenizer,
    BertPreTrainedModel,
    BertModel,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback, # Callback 기능을 위해 import
)
import numpy as np

# --- 1. Argument Parser 설정 ---
parser = argparse.ArgumentParser(description="Slot-Filling 모델 학습")
parser.add_argument("--train_file", type=str, default="train_with_literals.jsonl", help="훈련 데이터 파일 경로")
parser.add_argument("--eval_file", type=str, default="test_with_literals.jsonl", help="평가 데이터 파일 경로")
parser.add_argument("--model_name", type=str, default="bert-base-cased", help="사용할 모델 이름")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률")
parser.add_argument("--num_epochs", type=int, default=5, help="학습 에포크 수")
parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
parser.add_argument("--max_length", type=int, default=512, help="최대 시퀀스 길이")
parser.add_argument("--output_dir", type=str, default="./results_conditional", help="모델 저장 디렉토리")
parser.add_argument("--logging_dir", type=str, default="./logs_conditional", help="로그 저장 디렉토리")
parser.add_argument("--results_dir", type=str, default="./results", help="결과 저장 디렉토리")
parser.add_argument("--save_model_dir", type=str, default="./my_custom_trained_model", help="최종 모델 저장 디렉토리")
parser.add_argument("--num_test_examples", type=int, default=10000, help="추론 테스트할 예시 수")

args = parser.parse_args()

# --- 2. 설정 및 하이퍼파라미터 ---
class CFG:
    MODEL_NAME = args.model_name
    LEARNING_RATE = args.learning_rate
    NUM_TRAIN_EPOCHS = args.num_epochs
    PER_DEVICE_TRAIN_BATCH_SIZE = args.batch_size
    PER_DEVICE_EVAL_BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    OUTPUT_DIR = args.output_dir
    LOGGING_DIR = args.logging_dir
    LITERAL_PLACEHOLDER = "[LITERAL]"

# --- 3. 실제 데이터 로드 및 전처리 준비 (수정된 버전) ---

# 훈련 데이터와 평가(검증) 데이터 파일 경로를 각각 지정합니다.
train_file_path = args.train_file
eval_file_path = args.eval_file

# --- 훈련 데이터 로드 및 필터링 ---
print(f"훈련 데이터 로드: '{train_file_path}'")
try:
    # Hugging Face datasets 라이브러리를 사용하여 훈련 jsonl 파일 로드
    train_raw_dataset = datasets.load_dataset("json", data_files=train_file_path)['train']
except FileNotFoundError:
    print(f"에러: 훈련 파일 '{train_file_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

print(f"총 {len(train_raw_dataset)}개의 훈련 데이터를 로드했습니다.")

# 요구사항에 맞게 훈련 데이터 필터링 (label==1 and LITERAL_C > 0)
print("훈련 데이터를 필터링합니다...")
train_raw_dataset = train_raw_dataset.filter(
    lambda x: x.get('LITERAL_C', 0) > 0,
    num_proc=4 # 여러 프로세스를 사용하여 필터링 속도 향상
)
print(f"필터링 후 남은 훈련 데이터 수: {len(train_raw_dataset)}")


# --- 평가 데이터 로드 및 필터링 ---
print(f"\n평가 데이터 로드: '{eval_file_path}'")
try:
    # Hugging Face datasets 라이브러리를 사용하여 평가 jsonl 파일 로드
    eval_raw_dataset = datasets.load_dataset("json", data_files=eval_file_path)['train']
except FileNotFoundError:
    print(f"에러: 평가 파일 '{eval_file_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

print(f"총 {len(eval_raw_dataset)}개의 평가 데이터를 로드했습니다.")

# 요구사항에 맞게 평가 데이터 필터링 (label==0 and LITERAL_C > 0)
print("평가 데이터를 필터링합니다...")
eval_raw_dataset = eval_raw_dataset.filter(
    lambda x: x.get('label') == 0 and x.get('LITERAL_C', 0) > 0,
    num_proc=4
)
print(f"필터링 후 남은 평가 데이터 수: {len(eval_raw_dataset)}")

print("-" * 20)
print(f"최종 훈련 데이터 수: {len(train_raw_dataset)}")
print(f"최종 평가 데이터 수: {len(eval_raw_dataset)}")
print("-" * 20)

# --- 3. 토크나이저 및 전처리 함수 정의 ---
print("토크나이저 로드 및 전처리 함수를 정의합니다...")
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
tokenizer.add_special_tokens({'additional_special_tokens': [CFG.LITERAL_PLACEHOLDER]})
LITERAL_TOKEN_ID = tokenizer.convert_tokens_to_ids(CFG.LITERAL_PLACEHOLDER)

def preprocess_and_flatten_function(examples):
    questions = [q.replace("'[LITERAL]'", CFG.LITERAL_PLACEHOLDER).replace("[LITERAL]", CFG.LITERAL_PLACEHOLDER) for q in examples["masked_cypher"]]
    contexts = examples["nl_question"]
    tokenized_inputs = tokenizer(
        questions, contexts, max_length=CFG.MAX_LENGTH, padding="max_length",
        truncation=True, return_offsets_mapping=True
    )
    all_input_ids, all_attention_masks, all_mask_token_indices = [], [], []
    all_start_positions, all_end_positions = [], []
    for i in range(len(questions)):
        input_ids, sequence_ids, offset_mapping, answers = tokenized_inputs['input_ids'][i], tokenized_inputs.sequence_ids(batch_index=i), tokenized_inputs['offset_mapping'][i], examples["answers"][i]
        mask_indices_in_sequence = [idx for idx, token_id in enumerate(input_ids) if token_id == LITERAL_TOKEN_ID]
        for mask_order, mask_token_idx in enumerate(mask_indices_in_sequence):
            if mask_order >= len(answers): continue
            answer = answers[mask_order]
            start_char, end_char = answer["start_char"], answer["start_char"] + len(answer["text"])
            token_start_index, token_end_index = 0, len(input_ids) - 1
            while sequence_ids[token_start_index] != 1: token_start_index += 1
            while sequence_ids[token_end_index] != 1: token_end_index -= 1
            context_start_token, context_end_token = -1, -1
            for j, offset in enumerate(offset_mapping):
                if sequence_ids[j] == 1 and offset and offset[0] <= start_char < offset[1]: context_start_token = j
                if sequence_ids[j] == 1 and offset and offset[0] < end_char <= offset[1]: context_end_token = j
            if context_start_token != -1 and context_end_token != -1:
                all_input_ids.append(input_ids); all_attention_masks.append(tokenized_inputs['attention_mask'][i]); all_mask_token_indices.append(mask_token_idx)
                all_start_positions.append(context_start_token); all_end_positions.append(context_end_token)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "mask_token_indices": all_mask_token_indices,
            "start_positions": all_start_positions, "end_positions": all_end_positions}

print("데이터셋 전처리를 실행합니다...")
train_dataset = train_raw_dataset.map(preprocess_and_flatten_function, batched=True, remove_columns=train_raw_dataset.column_names, num_proc=4)
eval_dataset = eval_raw_dataset.map(preprocess_and_flatten_function, batched=True, remove_columns=eval_raw_dataset.column_names, num_proc=4)


# --- 4. 커스텀 모델 및 추론 함수 정의 (수정된 버전) ---
print("커스텀 모델 및 추론 함수를 정의합니다...")
class ConditionalSpanPredictor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        # 중요: 시작과 끝을 예측하는 분류기를 별도로 정의합니다.
        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                mask_token_indices=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        batch_size, seq_len, hidden_dim = sequence_output.shape
        
        idx = mask_token_indices.view(batch_size, 1, 1)
        idx_expanded = idx.expand(batch_size, 1, hidden_dim)
        mask_embedding = torch.gather(sequence_output, 1, idx_expanded) # [batch_size, 1, hidden_size]

        # --- [핵심 수정 부분: Bilinear Attention 방식 적용] ---
        # 1. 지문 토큰들을 시작점/끝점 예측을 위해 각각 변환(project)합니다.
        projected_start = self.start_classifier(sequence_output) # [batch_size, seq_len, hidden_size]
        projected_end = self.end_classifier(sequence_output)     # [batch_size, seq_len, hidden_size]

        # 2. 변환된 지문 토큰들과 [MASK] 임베딩을 행렬 곱셈(batch-wise dot product)하여 매칭 점수를 계산합니다.
        #    이것이 각 [MASK]에 대한 start/end logit이 됩니다.
        #    (projected_context) @ (mask_embedding.transpose)
        start_logits = torch.bmm(projected_start, mask_embedding.transpose(1, 2)).squeeze(-1) # [batch_size, seq_len]
        end_logits = torch.bmm(projected_end, mask_embedding.transpose(1, 2)).squeeze(-1)   # [batch_size, seq_len]
        # -----------------------------------------------------

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
        return {"loss": total_loss, "start_logits": start_logits, "end_logits": end_logits}

def predict_spans(question, context, model, tokenizer):
    device = model.device
    question = question.replace("'[LITERAL]'", CFG.LITERAL_PLACEHOLDER).replace("[LITERAL]", CFG.LITERAL_PLACEHOLDER)
    
    # --- [수정] return_offsets_mapping=True 추가 ---
    inputs = tokenizer(
        question, context, return_tensors='pt', 
        max_length=CFG.MAX_LENGTH, padding='max_length', truncation=True,
        return_offsets_mapping=True # 오프셋 정보를 얻기 위해 필수!
    )
    # offset_mapping 정보 가져오기
    offset_mapping = inputs.pop("offset_mapping")[0] # batch 차원 제거
    # ---------------------------------------------

    input_ids_list = inputs['input_ids'][0].tolist()
    mask_indices = [i for i, token_id in enumerate(input_ids_list) if token_id == LITERAL_TOKEN_ID]
    
    all_answers = []
    for mask_idx in mask_indices:
        model_inputs = inputs.copy()
        model_inputs['mask_token_indices'] = torch.tensor([mask_idx]).unsqueeze(0)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**model_inputs)
        
        start_pred_idx = outputs['start_logits'].argmax(dim=-1).item()
        end_pred_idx = outputs['end_logits'].argmax(dim=-1).item()
        
        # --- [수정] tokenizer.decode() 대신 오프셋으로 원본 텍스트 슬라이싱 ---
        if start_pred_idx > end_pred_idx:
            answer_text = "" # 시작점이 끝점보다 뒤면 빈 문자열 처리
        else:
            # 예측된 시작 토큰의 시작 문자 위치
            char_start_index = offset_mapping[start_pred_idx][0]
            # 예측된 끝 토큰의 끝 문자 위치
            char_end_index = offset_mapping[end_pred_idx][1]
            
            # 원본 context에서 해당 부분만 정확히 잘라냄
            answer_text = context[char_start_index:char_end_index]
        # -----------------------------------------------------------------

        all_answers.append(answer_text)
        
    return all_answers

# --- [새로운 기능] 커스텀 Trainer 정의 (업그레이드 버전) ---
class CustomTrainer(Trainer):
    def __init__(self, *args, eval_dataset_raw=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 원본 평가 데이터셋을 클래스 내에 저장
        self.eval_dataset_raw = eval_dataset_raw

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # 1. 부모 클래스(Trainer)의 기본 evaluate 메소드 호출하여 기본 메트릭(loss 등) 계산
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # 2. 우리가 원하는 '완전 일치 정확도' 수동 계산
        print("\n--- [평가] 완전 일치 정확도(Exact Match Ratio) 계산 중 ---")
        
        # --- [수정] LITERAL_C 개수별로 카운트를 저장할 딕셔너리 초기화 ---
        correct_counts_by_c = {}
        total_counts_by_c = {}
        
        for example in self.eval_dataset_raw:
            question = example["masked_cypher"]
            context = example["nl_question"]
            ground_truths = [ans['text'] for ans in example['answers']]
            
            # [LITERAL] 개수(LITERAL_C)를 가져옵니다.
            literal_c = example.get("LITERAL_C", 0)
            
            # 해당 LITERAL_C의 전체 카운트를 1 증가시킵니다.
            total_counts_by_c[literal_c] = total_counts_by_c.get(literal_c, 0) + 1
            
            predictions = predict_spans(question, context, self.model, self.tokenizer)
            
            # 예측과 정답이 일치하는지 확인
            if predictions == ground_truths:
                # 일치하면, 해당 LITERAL_C의 정답 카운트를 1 증가시킵니다.
                correct_counts_by_c[literal_c] = correct_counts_by_c.get(literal_c, 0) + 1
        
        # 3. 전체 EMR 계산 및 metrics 딕셔너리에 추가
        total_correct = sum(correct_counts_by_c.values())
        total_examples = sum(total_counts_by_c.values())
        overall_exact_match_ratio = total_correct / total_examples if total_examples > 0 else 0
        metrics[f"{metric_key_prefix}_exact_match"] = overall_exact_match_ratio
        
        print(f"--- Epoch {int(self.state.epoch)} Overall EMR: {overall_exact_match_ratio:.4f} ({total_correct}/{total_examples}) ---")

        # --- [추가] LITERAL_C 개수별 EMR 계산 및 출력/저장 ---
        print("--- [평가] LITERAL_C 개수별 EMR ---")
        # LITERAL_C 값으로 정렬하여 출력
        for c in sorted(total_counts_by_c.keys()):
            correct = correct_counts_by_c.get(c, 0)
            total = total_counts_by_c[c]
            ratio = correct / total if total > 0 else 0
            
            # 이 값을 트레이너 로그에 추가 (예: eval_emr_c1, eval_emr_c2)
            metrics[f"{metric_key_prefix}_emr_c{c}"] = ratio
            # 콘솔에 출력
            print(f"  - C={c}: {ratio:.4f} ({correct}/{total})")
        print("-" * 35)

        return metrics
    
if __name__ == "__main__":
    # --- 5. 훈련 설정 및 트레이너 정의 ---
    print("훈련을 위한 설정을 준비합니다...")
    model = ConditionalSpanPredictor.from_pretrained(CFG.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=CFG.OUTPUT_DIR,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=CFG.PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=CFG.NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=CFG.LOGGING_DIR,
        # 이제 'eval_exact_match'가 정상적으로 계산되므로, 이 기능을 다시 활성화합니다.
        load_best_model_at_end=True,
        metric_for_best_model="eval_exact_match", # CustomTrainer가 반환하는 키 이름
        greater_is_better=True, 
        report_to="none",
        save_total_limit=1
    )

    # [수정] 기존 Trainer 대신 CustomTrainer를 사용합니다.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        # 원본 평가셋을 전달하여 정확도 계산에 사용하도록 합니다.
        eval_dataset_raw=eval_raw_dataset 
    )

    # --- 6. 모델 훈련 ---
    print("모델 훈련을 시작합니다...")
    trainer.train()
    # load_best_model_at_end=True이므로, 훈련 후 가장 좋았던 모델이 로드된 상태
    trainer.save_model(args.save_model_dir)
    print("모델 훈련 및 저장 완료!")

    # --- 7. 추론 테스트 및 결과 저장 ---
    print(f"\n--- 훈련된 모델로 추론 테스트 (평가셋 상위 {args.num_test_examples}개) ---")

    # 최종적으로 가장 성능이 좋았던 모델을 다시 불러옵니다.
    final_model = ConditionalSpanPredictor.from_pretrained(args.save_model_dir)
    final_tokenizer = AutoTokenizer.from_pretrained(args.save_model_dir)

    # --- [수정] 결과 저장을 위한 파일 이름 지정 및 루프 카운트 변경 ---
    output_filename = f"{args.results_dir}/inference_results.txt"
    num_test_examples = min(args.num_test_examples, len(eval_raw_dataset))

    # 파일에 결과를 저장하기 위해 'w'(쓰기) 모드로 파일을 엽니다.
    with open(output_filename, 'w', encoding='utf-8') as f:
        print(f"추론 결과를 '{output_filename}' 파일에 저장합니다...")
        
        for i in range(num_test_examples):
            test_example = eval_raw_dataset[i]
            test_question = test_example["masked_cypher"]
            test_context = test_example["nl_question"]
            ground_truth_answers = [ans['text'] for ans in test_example['answers']]

            predicted_answers = predict_spans(test_question, test_context, final_model, final_tokenizer)

            # 출력을 위한 문자열 포맷팅
            result_str = (
                f"\n--- [추론 테스트 #{i+1}] ---\n"
                f"질문: {test_question}\n"
                f"지문: {test_context}\n"
                f"정답: {ground_truth_answers}\n"
                f"추론: {predicted_answers}\n"
                f"일치 여부: {'✅' if predicted_answers == ground_truth_answers else '❌'}\n"
            )
            
            # 파일에 결과 쓰기
            f.write(result_str)

    print(f"\n추론 결과 {num_test_examples}개가 '{output_filename}' 파일에 성공적으로 저장되었습니다.")