import json
import argparse
import torch
import os
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
import torch.nn as nn
from tqdm import tqdm
import re

# --- 1. 설정 및 커스텀 모델/함수 정의 (학습 때와 동일) ---

class CFG:
    MAX_LENGTH = 512
    LITERAL_PLACEHOLDER = "[LITERAL]"

class ConditionalSpanPredictor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
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
        mask_embedding = torch.gather(sequence_output, 1, idx_expanded)
        projected_start = self.start_classifier(sequence_output)
        projected_end = self.end_classifier(sequence_output)
        start_logits = torch.bmm(projected_start, mask_embedding.transpose(1, 2)).squeeze(-1)
        end_logits = torch.bmm(projected_end, mask_embedding.transpose(1, 2)).squeeze(-1)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        return {"loss": total_loss, "start_logits": start_logits, "end_logits": end_logits}

def predict_spans(question, context, model, tokenizer):
    device = model.device
    LITERAL_TOKEN_ID = tokenizer.convert_tokens_to_ids(CFG.LITERAL_PLACEHOLDER)
    question = question.replace("'[LITERAL]'", CFG.LITERAL_PLACEHOLDER).replace("[LITERAL]", CFG.LITERAL_PLACEHOLDER)
    inputs = tokenizer(
        question, context, return_tensors='pt', 
        max_length=CFG.MAX_LENGTH, padding='max_length', truncation=True,
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
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
        if start_pred_idx > end_pred_idx or start_pred_idx >= len(offset_mapping) or end_pred_idx >= len(offset_mapping):
            answer_text = ""
        else:
            char_start_index = offset_mapping[start_pred_idx][0]
            char_end_index = offset_mapping[end_pred_idx][1]
            answer_text = context[char_start_index:char_end_index]
        all_answers.append(answer_text)
    return all_answers

# --- 2. 메인 실행 부분 (수정됨) ---
def run_inference(args):
    print("추론 및 평가를 시작합니다...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    print(f"모델 로딩: {args.model_path}")
    model = ConditionalSpanPredictor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print(f"입력 데이터 로딩: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # ▼▼▼ 'label'이 0인 데이터만 필터링하는 코드 추가 ▼▼▼
    original_data_count = len(data)
    data = [item for item in data if item.get('label') == 0]
    print(f"필터링 후 남은 데이터 수 (label == 0): {len(data)}개 ({original_data_count - len(data)}개 제외)")

    num_examples = len(data)
    if args.num_examples is not None:
        num_examples = min(args.num_examples, len(data))

    # 카운터 초기화
    span_correct_count = 0  # Slot-filling 정확도
    exec_correct_count = 0  # 최종 쿼리 일치 정확도
    
    # 쿼리 비교를 위한 정규화 함수
    normalize = lambda s: re.sub(r'\s+','', s.lower())

    print(f"추론 결과를 '{args.output_txt_file}' 와 '{args.output_jsonl_file}' 파일에 저장합니다...")
    with open(args.output_txt_file, 'w', encoding='utf-8') as f_txt, \
         open(args.output_jsonl_file, 'w', encoding='utf-8') as f_jsonl:
        
        for i in tqdm(range(num_examples), desc="추론 진행 중"):
            example = data[i]
            masked_cypher = example["masked_cypher"]
            nl_question = example["nl_question"]
            ground_truth_answers = [ans['text'] for ans in example['answers']]
            gold_cypher = example["gold_cypher"]
            
            # 1. Slot Filling 예측
            predicted_answers = predict_spans(masked_cypher, nl_question, model, tokenizer)
            
            # ▼▼▼ 여기가 수정된 핵심 부분입니다 ▼▼▼
            # 예측된 값으로 Cypher 쿼리 재구성
            predicted_cypher = masked_cypher
            for answer in predicted_answers:
                # 추가로 따옴표를 씌우지 않고, 예측된 answer를 그대로 사용합니다.
                predicted_cypher = predicted_cypher.replace(CFG.LITERAL_PLACEHOLDER, answer, 1)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


            # 3. 정확도 계산
            is_span_correct = (predicted_answers == ground_truth_answers)
            if is_span_correct:
                span_correct_count += 1
            
            is_exec_correct = (normalize(predicted_cypher) == normalize(gold_cypher))
            if is_exec_correct:
                exec_correct_count += 1


            # ▼▼▼ 추가할 디버깅 코드 ▼▼▼
            # Slot-Filling(EM)은 틀렸지만, 실행 쿼리(EA)는 맞은 경우를 출력
            if not is_span_correct and is_exec_correct:
                print("\n" + "="*50)
                print("✅ DISCREPANCY FOUND (EA > EM)")
                print(f"질문: {nl_question}")
                print(f"정답 리터럴: {ground_truth_answers}")
                print(f"추론 리터럴: {predicted_answers}  <-- 리스트는 다르지만")
                print(f"GOLD CYPHER: {gold_cypher}")
                print(f"PRED CYPHER: {predicted_cypher} <-- 정규화하면 같아짐")
                print("="*50 + "\n")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            
            # 4. .txt 파일에 상세 로그 저장 (기존 기능 유지)
            result_str = (
                f"\n--- [추론 테스트 #{i+1}] ---\n"
                f"질문: {nl_question}\n"
                f"정답 리터럴: {ground_truth_answers}\n"
                f"추론 리터럴: {predicted_answers}\n"
                f"리터럴 일치 여부: {'✅' if is_span_correct else '❌'}\n"
                f"GOLD CYPHER : {gold_cypher}\n"
                f"PRED CYPHER : {predicted_cypher}\n"
                f"쿼리 일치 여부: {'✅' if is_exec_correct else '❌'}\n"
            )
            f_txt.write(result_str)

            # 5. .jsonl 파일에 최종 결과 저장
            json_output = {
                "nlq": nl_question,
                "gold_cypher": gold_cypher,
                "predicted_cypher": predicted_cypher
            }
            f_jsonl.write(json.dumps(json_output, ensure_ascii=False) + '\n')

        # --- 최종 결과 요약 ---
        span_accuracy = (span_correct_count / num_examples) * 100 if num_examples > 0 else 0
        exec_accuracy = (exec_correct_count / num_examples) * 100 if num_examples > 0 else 0
        
        summary_str = (
            f"\n\n==================== 최종 결과 요약 ====================\n"
            f"Slot Filling 정확도 (EM): {span_accuracy:.2f}% ({span_correct_count} / {num_examples})\n"
            f"실행 쿼리 정확도 (EA)  : {exec_accuracy:.2f}% ({exec_correct_count} / {num_examples})\n"
            f"========================================================\n"
        )
        print(summary_str)
        f_txt.write(summary_str)

    print(f"\n추론 및 평가 완료! 결과가 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .jsonl file for inference")
    parser.add_argument("--output_txt_file", type=str, required=True, help="Path to save the output .txt file")
    parser.add_argument("--output_jsonl_file", type=str, required=True, help="Path to save the machine-readable .jsonl file")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to run inference on. Defaults to all.")

    args = parser.parse_args()
    run_inference(args)


# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --model_path ./my_custom_trained_model \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./final_predictions.txt \
#     --output_jsonl_file ./final_predictions.json
