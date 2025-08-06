import json
import argparse
import torch
import os
import time
import csv
import psutil
import gc
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import re
from collections import defaultdict

# --- 1. 설정 및 T5 기반 Slot Filling 모델 정의 ---

class CFG:
    MAX_LENGTH = 512
    LITERAL_PLACEHOLDER = "[LITERAL]"

def get_gpu_memory_usage():
    """GPU 메모리 사용량 반환 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_system_memory_usage():
    """시스템 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class T5SlotFillingModel:
    def __init__(self, model_path=None, pretrained_model="t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 메모리 측정 시작
        initial_gpu_memory = get_gpu_memory_usage()
        initial_system_memory = get_system_memory_usage()
        
        # model_path가 있으면 로컬 모델, 없으면 HF pretrained 모델 사용
        if model_path and os.path.exists(model_path):
            print(f"로컬 모델 로딩: {model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            print(f"Hugging Face pretrained 모델 로딩: {pretrained_model}")
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        
        self.model.to(self.device)
        self.model.eval()
        
        # T5 토크나이저에 특수 토큰 추가
        if CFG.LITERAL_PLACEHOLDER not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([CFG.LITERAL_PLACEHOLDER])
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 메모리 측정 완료
        final_gpu_memory = get_gpu_memory_usage()
        final_system_memory = get_system_memory_usage()
        
        # 모델 로딩으로 인한 메모리 증가량 계산
        self.model_gpu_memory = final_gpu_memory - initial_gpu_memory
        self.model_system_memory = final_system_memory - initial_system_memory
        
        print(f"모델 로딩 후 GPU 메모리 증가량: {self.model_gpu_memory:.2f} MB")
        print(f"모델 로딩 후 시스템 메모리 증가량: {self.model_system_memory:.2f} MB")
    
    def get_memory_info(self):
        """현재 메모리 정보 반환"""
        current_gpu_memory = get_gpu_memory_usage()
        current_system_memory = get_system_memory_usage()
        
        return {
            'current_gpu_memory': current_gpu_memory,
            'current_system_memory': current_system_memory,
            'model_gpu_memory': self.model_gpu_memory,
            'model_system_memory': self.model_system_memory
        }
    
    def create_input_text(self, masked_cypher, nl_question):
        """T5 입력 텍스트 생성"""
        # 입력 형식: "Question: {nl_question} Template: {masked_cypher}"
        input_text = f"Question: {nl_question} Template: {masked_cypher}"
        return input_text
    
    def create_target_text(self, answers):
        """T5 타겟 텍스트 생성"""
        # 출력 형식: "Answer1: {answer1} Answer2: {answer2} ..."
        target_parts = []
        for i, answer in enumerate(answers):
            target_parts.append(f"Answer{i+1}: {answer}")
        return " ".join(target_parts)
    
    def parse_target_text(self, target_text):
        """T5 출력 텍스트에서 답변 추출"""
        answers = []
        
        # 더 간단하고 확실한 방법으로 파싱
        # "Answer1: value1 Answer2: value2" 형식에서 값 추출
        parts = target_text.split('Answer')
        
        for part in parts[1:]:  # 첫 번째 빈 부분 제외
            if ':' in part:
                # "1: value1 Answer2: value2" -> "1: value1" 부분만 추출
                colon_pos = part.find(':')
                if colon_pos != -1:
                    # 숫자 부분 제거하고 값만 추출
                    value_part = part[colon_pos + 1:].strip()
                    # 다음 Answer가 있으면 그 전까지만
                    next_answer_pos = value_part.find('Answer')
                    if next_answer_pos != -1:
                        value_part = value_part[:next_answer_pos].strip()
                    answers.append(value_part)
        
        # 디버깅을 위한 출력
        print(f"DEBUG - 원본 텍스트: '{target_text}'")
        print(f"DEBUG - 파싱된 답변: {answers}")
        
        return answers
    
    def predict_spans_batch(self, masked_cyphers, nl_questions):
        """T5를 사용하여 배치 단위로 슬롯 채우기 예측"""
        # 배치 입력 텍스트 생성
        input_texts = []
        for masked_cypher, nl_question in zip(masked_cyphers, nl_questions):
            input_text = self.create_input_text(masked_cypher, nl_question)
            input_texts.append(input_text)
        
        # 배치 토크나이징
        inputs = self.tokenizer(
            input_texts,
            max_length=CFG.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # T5 배치 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=CFG.MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        # 배치 출력 디코딩
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 각 텍스트에서 답변 추출 및 토큰 수 계산
        all_predicted_answers = []
        token_counts = []
        for i, (generated_text, masked_cypher) in enumerate(zip(generated_texts, masked_cyphers)):
            predicted_answers = self.parse_target_text(generated_text)
            
            # [LITERAL] 토큰 개수만큼 답변 생성
            literal_count = masked_cypher.count(CFG.LITERAL_PLACEHOLDER)
            if len(predicted_answers) < literal_count:
                # 부족한 답변은 빈 문자열로 채움
                predicted_answers.extend([""] * (literal_count - len(predicted_answers)))
            elif len(predicted_answers) > literal_count:
                # 초과한 답변은 제거
                predicted_answers = predicted_answers[:literal_count]
            
            all_predicted_answers.append(predicted_answers)
            token_counts.append(len(outputs[i]))  # 각 배치 항목의 토큰 수
        
        return all_predicted_answers, token_counts
    
    def predict_spans_single(self, masked_cypher, nl_question):
        """단일 예측 (Latency 측정용)"""
        input_text = self.create_input_text(masked_cypher, nl_question)
        
        # 입력 토크나이징
        inputs = self.tokenizer(
            input_text,
            max_length=CFG.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # T5 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=CFG.MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        # 출력 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 생성된 텍스트에서 답변 추출
        predicted_answers = self.parse_target_text(generated_text)
        
        # [LITERAL] 토큰 개수만큼 답변 생성
        literal_count = masked_cypher.count(CFG.LITERAL_PLACEHOLDER)
        if len(predicted_answers) < literal_count:
            # 부족한 답변은 빈 문자열로 채움
            predicted_answers.extend([""] * (literal_count - len(predicted_answers)))
        elif len(predicted_answers) > literal_count:
            # 초과한 답변은 제거
            predicted_answers = predicted_answers[:literal_count]
        
        return predicted_answers, len(outputs[0]), generated_text  # 토큰 수도 반환
    
    def predict_spans(self, masked_cypher, nl_question):
        """단일 예측 (기존 호환성 유지)"""
        result, _ = self.predict_spans_single(masked_cypher, nl_question)
        return result

# --- 2. 성능 측정 클래스 ---
class PerformanceMetrics:
    def __init__(self):
        self.latencies = []
        self.batch_times = []
        self.total_queries = 0
        self.start_time = None
        self.end_time = None
        self.all_latencies = []  # 모든 쿼리의 latency 저장
        self.token_counts = []  # 각 쿼리별 생성된 토큰 수
        self.peak_gpu_memory = 0
        self.peak_system_memory = 0
        self.memory_samples = []  # 메모리 샘플링 데이터
    
    def start_measurement(self):
        """전체 측정 시작"""
        self.start_time = time.time()
        # 초기 메모리 측정
        self.peak_gpu_memory = get_gpu_memory_usage()
        self.peak_system_memory = get_system_memory_usage()
    
    def end_measurement(self):
        """전체 측정 종료"""
        self.end_time = time.time()
    
    def add_latency(self, latency, token_count=None):
        """단일 쿼리 Latency 추가"""
        self.latencies.append(latency)
        self.all_latencies.append(latency)  # 모든 latency 저장
        self.total_queries += 1
        
        if token_count is not None:
            self.token_counts.append(token_count)
        
        # 메모리 샘플링
        current_gpu = get_gpu_memory_usage()
        current_system = get_system_memory_usage()
        self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu)
        self.peak_system_memory = max(self.peak_system_memory, current_system)
        self.memory_samples.append((current_gpu, current_system))
    
    def add_batch_time(self, batch_time, batch_size, token_counts=None):
        """배치 처리 시간 추가"""
        self.batch_times.append((batch_time, batch_size))
        # 배치당 평균 latency를 batch_size만큼 저장
        avg_latency = batch_time / batch_size
        for _ in range(batch_size):
            self.all_latencies.append(avg_latency)
        self.total_queries += batch_size
        
        if token_counts:
            self.token_counts.extend(token_counts)
        
        # 메모리 샘플링
        current_gpu = get_gpu_memory_usage()
        current_system = get_system_memory_usage()
        self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu)
        self.peak_system_memory = max(self.peak_system_memory, current_system)
        self.memory_samples.append((current_gpu, current_system))
    
    def get_metrics(self):
        """성능 지표 계산"""
        if not self.latencies and not self.batch_times:
            return {}
        
        metrics = {}
        
        # Latency 통계 (단일 쿼리)
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            metrics['single_latency'] = {
                'mean': sum(self.latencies) / n,
                'median': sorted_latencies[n // 2] if n % 2 == 1 else (sorted_latencies[n // 2 - 1] + sorted_latencies[n // 2]) / 2,
                'q1': sorted_latencies[n // 4],
                'q3': sorted_latencies[3 * n // 4],
                'min': min(self.latencies),
                'max': max(self.latencies),
                'count': n
            }
        
        # Throughput 계산 (실제 추론 시간만 고려)
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            metrics['overall_throughput'] = self.total_queries / total_time
        
        # 배치 처리 통계 및 정확한 QPS 계산
        if self.batch_times:
            batch_latencies = []
            total_batch_time = 0
            total_batch_queries = 0
            
            for batch_time, batch_size in self.batch_times:
                avg_latency = batch_time / batch_size
                batch_latencies.extend([avg_latency] * batch_size)
                total_batch_time += batch_time
                total_batch_queries += batch_size
            
            # 배치 처리의 실제 QPS 계산 (배치 크기 고려)
            if total_batch_time > 0:
                # 실제 배치 처리 QPS: 배치당 처리량
                metrics['batch_throughput'] = total_batch_queries / total_batch_time
                
                # 배치별 평균 QPS (더 정확한 측정)
                batch_qps_list = []
                for batch_time, batch_size in self.batch_times:
                    if batch_time > 0:
                        batch_qps = batch_size / batch_time
                        batch_qps_list.append(batch_qps)
                
                if batch_qps_list:
                    metrics['avg_batch_qps'] = sum(batch_qps_list) / len(batch_qps_list)
                    metrics['min_batch_qps'] = min(batch_qps_list)
                    metrics['max_batch_qps'] = max(batch_qps_list)
            
            sorted_batch_latencies = sorted(batch_latencies)
            n = len(sorted_batch_latencies)
            metrics['batch_latency'] = {
                'mean': sum(batch_latencies) / n,
                'median': sorted_batch_latencies[n // 2] if n % 2 == 1 else (sorted_batch_latencies[n // 2 - 1] + sorted_batch_latencies[n // 2]) / 2,
                'q1': sorted_batch_latencies[n // 4],
                'q3': sorted_batch_latencies[3 * n // 4],
                'min': min(batch_latencies),
                'max': max(batch_latencies),
                'count': n
            }
        
        # 메모리 통계
        if self.memory_samples:
            gpu_memories = [sample[0] for sample in self.memory_samples]
            system_memories = [sample[1] for sample in self.memory_samples]
            
            metrics['memory'] = {
                'peak_gpu_memory': self.peak_gpu_memory,
                'peak_system_memory': self.peak_system_memory,
                'avg_gpu_memory': sum(gpu_memories) / len(gpu_memories),
                'avg_system_memory': sum(system_memories) / len(system_memories),
                'min_gpu_memory': min(gpu_memories),
                'min_system_memory': min(system_memories),
                'max_gpu_memory': max(gpu_memories),
                'max_system_memory': max(system_memories)
            }
        
        # 토큰 수 통계
        if self.token_counts:
            metrics['token_stats'] = {
                'avg_tokens': sum(self.token_counts) / len(self.token_counts),
                'min_tokens': min(self.token_counts),
                'max_tokens': max(self.token_counts),
                'total_tokens': sum(self.token_counts)
            }
        
        return metrics

# --- 3. 메인 실행 부분 ---
def run_inference(args):
    print("T5 기반 Slot Filling 추론 및 평가를 시작합니다...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    print(f"배치 크기: {args.batch_size}")
    print(f"성능 측정: {'활성화' if args.measure_performance else '비활성화'}")

    print(f"T5 모델 로딩: {args.model_path or args.pretrained_model}")
    model = T5SlotFillingModel(args.model_path, args.pretrained_model)
    print("모델 로딩 완료!")

    print(f"입력 데이터 로딩: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 'label'이 0인 데이터만 필터링
    original_data_count = len(data)
    data = [item for item in data if item.get('label') == 0]
    print(f"필터링 후 남은 데이터 수 (label == 0): {len(data)}개 ({original_data_count - len(data)}개 제외)")

    num_examples = len(data)
    if args.num_examples is not None:
        num_examples = min(args.num_examples, len(data))

    # 성능 측정 초기화 (모델 로딩 완료 후 시작)
    performance_metrics = PerformanceMetrics()
    
    # 실제 추론 시작 시간 기록
    inference_start_time = time.time()
    print(f"실제 추론 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(inference_start_time))}")
    
    # 성능 측정 시작
    performance_metrics.start_measurement()

    # 카운터 초기화
    span_correct_count = 0  # Slot-filling 정확도
    exec_correct_count = 0  # 최종 쿼리 일치 정확도
    
    # 쿼리 비교를 위한 정규화 함수
    normalize = lambda s: re.sub(r'\s+','', s.lower())

    print(f"추론 결과를 '{args.output_txt_file}' 와 '{args.output_jsonl_file}' 파일에 저장합니다...")
    with open(args.output_txt_file, 'w', encoding='utf-8') as f_txt, \
         open(args.output_jsonl_file, 'w', encoding='utf-8') as f_jsonl:
        
        if args.measure_performance and args.batch_size == 1:
            # 단일 쿼리 Latency 측정
            print("단일 쿼리 Latency 측정 모드...")
            for i in tqdm(range(num_examples), desc="T5 단일 추론 진행 중"):
                example = data[i]
                masked_cypher = example["masked_cypher"]
                nl_question = example["nl_question"]
                ground_truth_answers = [ans['text'] for ans in example['answers']]
                gold_cypher = example["gold_cypher"]
                
                # Latency 측정
                start_time = time.time()
                predicted_answers, token_count, generated_text = model.predict_spans_single(masked_cypher, nl_question)
                latency = time.time() - start_time
                performance_metrics.add_latency(latency, token_count)
                
                # 결과 처리
                predicted_cypher = masked_cypher
                for answer in predicted_answers:
                    predicted_cypher = predicted_cypher.replace(CFG.LITERAL_PLACEHOLDER, answer, 1)

                # 정확도 계산
                is_span_correct = (predicted_answers == ground_truth_answers)
                if is_span_correct:
                    span_correct_count += 1
                
                is_exec_correct = (normalize(predicted_cypher) == normalize(gold_cypher))
                if is_exec_correct:
                    exec_correct_count += 1

                # 디버깅 출력
                if not is_span_correct and is_exec_correct:
                    print("\n" + "="*50)
                    print("✅ T5 DISCREPANCY FOUND (EA > EM)")
                    print(f"질문: {nl_question}")
                    print(f"정답 리터럴: {ground_truth_answers}")
                    print(f"T5 추론 리터럴: {predicted_answers}")
                    print(f"GOLD CYPHER: {gold_cypher}")
                    print(f"PRED CYPHER: {predicted_cypher}")
                    print(f"Latency: {latency:.4f}초")
                    print("="*50 + "\n")
                
                # 결과 저장
                result_str = (
                    f"\n--- [T5 추론 테스트 #{i+1}] ---\n"
                    f"질문: {nl_question}\n"
                    f"정답 리터럴: {ground_truth_answers}\n"
                    f"T5 추론 리터럴: {predicted_answers}\n"
                    f"T5 원본 생성 텍스트: {generated_text}\n"
                    f"리터럴 일치 여부: {'✅' if is_span_correct else '❌'}\n"
                    f"GOLD CYPHER : {gold_cypher}\n"
                    f"PRED CYPHER : {predicted_cypher}\n"
                    f"쿼리 일치 여부: {'✅' if is_exec_correct else '❌'}\n"
                    f"Latency: {latency:.4f}초\n"
                    f"생성된 토큰 수: {token_count}\n"
                )
                f_txt.write(result_str)

                json_output = {
                    "nlq": nl_question,
                    "gold_cypher": gold_cypher,
                    "predicted_cypher": predicted_cypher,
                    "model_type": "T5",
                    "latency": latency,
                    "token_count": token_count,
                    "t5_generated_text": generated_text
                }
                f_jsonl.write(json.dumps(json_output, ensure_ascii=False) + '\n')
        
        else:
            # 배치 처리
            print("배치 처리 모드...")
            for batch_start in tqdm(range(0, num_examples, args.batch_size), desc="T5 배치 추론 진행 중"):
                batch_end = min(batch_start + args.batch_size, num_examples)
                batch_data = data[batch_start:batch_end]
                
                # 배치 데이터 준비
                masked_cyphers = [item["masked_cypher"] for item in batch_data]
                nl_questions = [item["nl_question"] for item in batch_data]
                ground_truth_answers_list = [[ans['text'] for ans in item['answers']] for item in batch_data]
                gold_cyphers = [item["gold_cypher"] for item in batch_data]
                
                # 배치 처리 시간 측정
                if args.measure_performance:
                    start_time = time.time()
                
                # T5 기반 Slot Filling 배치 예측
                predicted_answers_batch, token_counts_batch = model.predict_spans_batch(masked_cyphers, nl_questions)
                
                if args.measure_performance:
                    batch_time = time.time() - start_time
                    performance_metrics.add_batch_time(batch_time, len(batch_data), token_counts_batch)
                
                # 각 예시별로 결과 처리
                for i, (example, predicted_answers, ground_truth_answers, gold_cypher) in enumerate(
                    zip(batch_data, predicted_answers_batch, ground_truth_answers_list, gold_cyphers)
                ):
                    example_idx = batch_start + i
                    masked_cypher = example["masked_cypher"]
                    nl_question = example["nl_question"]
                    token_count = token_counts_batch[i] if args.measure_performance else None
                    
                    # 예측된 값으로 Cypher 쿼리 재구성
                    predicted_cypher = masked_cypher
                    for answer in predicted_answers:
                        predicted_cypher = predicted_cypher.replace(CFG.LITERAL_PLACEHOLDER, answer, 1)

                    # 정확도 계산
                    is_span_correct = (predicted_answers == ground_truth_answers)
                    if is_span_correct:
                        span_correct_count += 1
                    
                    is_exec_correct = (normalize(predicted_cypher) == normalize(gold_cypher))
                    if is_exec_correct:
                        exec_correct_count += 1

                    # 디버깅 출력
                    if not is_span_correct and is_exec_correct:
                        print("\n" + "="*50)
                        print("✅ T5 DISCREPANCY FOUND (EA > EM)")
                        print(f"질문: {nl_question}")
                        print(f"정답 리터럴: {ground_truth_answers}")
                        print(f"T5 추론 리터럴: {predicted_answers}")
                        print(f"GOLD CYPHER: {gold_cypher}")
                        print(f"PRED CYPHER: {predicted_cypher}")
                        if args.measure_performance:
                            print(f"배치 처리 시간: {batch_time:.4f}초 (배치 크기: {len(batch_data)})")
                            print(f"생성된 토큰 수: {token_count}")
                        print("="*50 + "\n")
                    
                    # 결과 저장
                    result_str = (
                        f"\n--- [T5 추론 테스트 #{example_idx+1}] ---\n"
                        f"질문: {nl_question}\n"
                        f"정답 리터럴: {ground_truth_answers}\n"
                        f"T5 추론 리터럴: {predicted_answers}\n"
                        f"리터럴 일치 여부: {'✅' if is_span_correct else '❌'}\n"
                        f"GOLD CYPHER : {gold_cypher}\n"
                        f"PRED CYPHER : {predicted_cypher}\n"
                        f"쿼리 일치 여부: {'✅' if is_exec_correct else '❌'}\n"
                    )
                    if args.measure_performance:
                        result_str += f"배치 처리 시간: {batch_time:.4f}초 (배치 크기: {len(batch_data)})\n"
                        result_str += f"생성된 토큰 수: {token_count}\n"
                    f_txt.write(result_str)

                    json_output = {
                        "nlq": nl_question,
                        "gold_cypher": gold_cypher,
                        "predicted_cypher": predicted_cypher,
                        "model_type": "T5"
                    }
                    if args.measure_performance:
                        json_output["batch_time"] = batch_time
                        json_output["batch_size"] = len(batch_data)
                        json_output["token_count"] = token_count
                    f_jsonl.write(json.dumps(json_output, ensure_ascii=False) + '\n')

    # 성능 측정 종료
    performance_metrics.end_measurement()
    metrics = performance_metrics.get_metrics()
    
    # 실제 추론 완료 시간 기록
    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    print(f"실제 추론 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(inference_end_time))}")
    print(f"총 추론 시간: {total_inference_time:.4f}초 ({total_inference_time/60:.2f}분)")

    # --- 최종 결과 요약 ---
    span_accuracy = (span_correct_count / num_examples) * 100 if num_examples > 0 else 0
    exec_accuracy = (exec_correct_count / num_examples) * 100 if num_examples > 0 else 0
    
    summary_str = (
        f"\n\n==================== T5 최종 결과 요약 ====================\n"
        f"배치 크기: {args.batch_size}\n"
        f"총 추론 시간: {total_inference_time:.4f}초 ({total_inference_time/60:.2f}분)\n"
        f"T5 Slot Filling 정확도 (EM): {span_accuracy:.2f}% ({span_correct_count} / {num_examples})\n"
        f"T5 실행 쿼리 정확도 (EA)  : {exec_accuracy:.2f}% ({exec_correct_count} / {num_examples})\n"
    )
    
    # 성능 지표 추가
    if args.measure_performance and metrics:
        summary_str += f"\n--- 성능 지표 ---\n"
        
        # QPS 계산 공식 설명
        summary_str += f"QPS 계산 공식:\n"
        if args.batch_size == 1:
            summary_str += f"  - 단일 쿼리 QPS = 1 / 평균 Latency\n"
            if 'single_latency' in metrics:
                theoretical_qps = 1 / metrics['single_latency']['mean']
                summary_str += f"  - 이론적 QPS = 1 / {metrics['single_latency']['mean']:.4f} = {theoretical_qps:.2f} queries/second\n"
        else:
            summary_str += f"  - 배치 QPS = 배치 크기 / 배치 처리 시간\n"
            summary_str += f"  - 전체 QPS = 총 쿼리 수 / 총 처리 시간\n"
        
        if 'overall_throughput' in metrics:
            summary_str += f"전체 Throughput: {metrics['overall_throughput']:.2f} queries/second\n"
        
        if 'batch_throughput' in metrics:
            summary_str += f"배치 처리 Throughput: {metrics['batch_throughput']:.2f} queries/second\n"
        
        if 'avg_batch_qps' in metrics:
            summary_str += f"배치별 평균 QPS: {metrics['avg_batch_qps']:.2f} queries/second\n"
            summary_str += f"배치별 최소 QPS: {metrics['min_batch_qps']:.2f} queries/second\n"
            summary_str += f"배치별 최대 QPS: {metrics['max_batch_qps']:.2f} queries/second\n"
        
        if 'single_latency' in metrics:
            summary_str += f"단일 쿼리 Latency:\n"
            summary_str += f"  - 평균: {metrics['single_latency']['mean']:.4f}초\n"
            summary_str += f"  - 중앙값: {metrics['single_latency']['median']:.4f}초\n"
            summary_str += f"  - Q1: {metrics['single_latency']['q1']:.4f}초\n"
            summary_str += f"  - Q3: {metrics['single_latency']['q3']:.4f}초\n"
            summary_str += f"  - 최소: {metrics['single_latency']['min']:.4f}초\n"
            summary_str += f"  - 최대: {metrics['single_latency']['max']:.4f}초\n"
        
        if 'batch_latency' in metrics:
            summary_str += f"배치 처리 Latency (평균):\n"
            summary_str += f"  - 평균: {metrics['batch_latency']['mean']:.4f}초\n"
            summary_str += f"  - 중앙값: {metrics['batch_latency']['median']:.4f}초\n"
            summary_str += f"  - Q1: {metrics['batch_latency']['q1']:.4f}초\n"
            summary_str += f"  - Q3: {metrics['batch_latency']['q3']:.4f}초\n"
            summary_str += f"  - 최소: {metrics['batch_latency']['min']:.4f}초\n"
            summary_str += f"  - 최대: {metrics['batch_latency']['max']:.4f}초\n"
        
        if 'memory' in metrics:
            summary_str += f"메모리 사용량:\n"
            summary_str += f"  - 최대 GPU: {metrics['memory']['peak_gpu_memory']:.2f} MB\n"
            summary_str += f"  - 최대 시스템: {metrics['memory']['peak_system_memory']:.2f} MB\n"
            summary_str += f"  - 평균 GPU: {metrics['memory']['avg_gpu_memory']:.2f} MB\n"
            summary_str += f"  - 평균 시스템: {metrics['memory']['avg_system_memory']:.2f} MB\n"
            summary_str += f"  - 최소 GPU: {metrics['memory']['min_gpu_memory']:.2f} MB\n"
            summary_str += f"  - 최소 시스템: {metrics['memory']['min_system_memory']:.2f} MB\n"
        
        if 'token_stats' in metrics:
            summary_str += f"토큰 수:\n"
            summary_str += f"  - 평균: {metrics['token_stats']['avg_tokens']:.2f}\n"
            summary_str += f"  - 최소: {metrics['token_stats']['min_tokens']:.2f}\n"
            summary_str += f"  - 최대: {metrics['token_stats']['max_tokens']:.2f}\n"
            summary_str += f"  - 총합: {metrics['token_stats']['total_tokens']:.2f}\n"
    
    summary_str += f"========================================================\n"
    print(summary_str)
    
    with open(args.output_txt_file, 'a', encoding='utf-8') as f_txt:
        f_txt.write(summary_str)

    # CSV 파일로 latency 데이터 저장
    if args.measure_performance and performance_metrics.all_latencies:
        csv_filename = args.output_txt_file.replace('.txt', '_latency.csv')
        save_latency_to_csv(performance_metrics.all_latencies, csv_filename, args.pretrained_model)
        print(f"Latency 데이터가 '{csv_filename}'에 저장되었습니다.")

    print(f"\nT5 기반 추론 및 평가 완료! 결과가 저장되었습니다.")

def save_latency_to_csv(latencies, csv_filename, model_name):
    """Latency 데이터를 CSV 파일로 저장"""
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 헤더 작성
        writer.writerow(['Query_Index', 'Latency_Seconds', 'Model'])
        
        # 데이터 작성
        for i, latency in enumerate(latencies):
            writer.writerow([i+1, latency, model_name])
    
    print(f"총 {len(latencies)}개의 latency 데이터가 저장되었습니다.")

# --- 4. 학습 데이터 생성 함수 (T5 학습용) ---
def create_t5_training_data(input_file, output_file):
    """T5 학습을 위한 데이터 생성"""
    print(f"T5 학습 데이터 생성 중: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    t5_data = []
    for item in data:
        if item.get('label') == 0:  # 유효한 데이터만
            masked_cypher = item["masked_cypher"]
            nl_question = item["nl_question"]
            ground_truth_answers = [ans['text'] for ans in item['answers']]
            
            # T5 입력/출력 형식으로 변환
            input_text = f"Question: {nl_question} Template: {masked_cypher}"
            target_text = " ".join([f"Answer{i+1}: {ans}" for i, ans in enumerate(ground_truth_answers)])
            
            t5_data.append({
                "input": input_text,
                "target": target_text
            })
    
    # T5 학습 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in t5_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"T5 학습 데이터 생성 완료: {len(t5_data)}개 샘플")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the T5 model directory (optional)")
    parser.add_argument("--pretrained_model", type=str, default="t5-base", help="Hugging Face pretrained model name (default: t5-base)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .jsonl file for inference")
    parser.add_argument("--output_txt_file", type=str, required=True, help="Path to save the output .txt file")
    parser.add_argument("--output_jsonl_file", type=str, required=True, help="Path to save the machine-readable .jsonl file")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to run inference on. Defaults to all.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--measure_performance", action="store_true", help="Measure latency and throughput")
    parser.add_argument("--create_training_data", action="store_true", help="Create T5 training data from input file")
    parser.add_argument("--training_data_output", type=str, help="Output file for T5 training data")

    args = parser.parse_args()
    
    if args.create_training_data:
        if not args.training_data_output:
            print("Error: --training_data_output is required when --create_training_data is used")
            exit(1)
        create_t5_training_data(args.input_file, args.training_data_output)
    else:
        run_inference(args)

# 사용 예시:
# 1. 기본 배치 처리 (성능 측정 없음):
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_batch_predictions.txt \
#     --output_jsonl_file ./t5_batch_predictions.json \
#     --batch_size 8

# 2. 성능 측정 포함 배치 처리:
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_performance_predictions.txt \
#     --output_jsonl_file ./t5_performance_predictions.json \
#     --batch_size 16 \
#     --measure_performance

# 3. 단일 쿼리 Latency 측정:
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_latency_predictions.txt \
#     --output_jsonl_file ./t5_latency_predictions.json \
#     --batch_size 1 \
#     --measure_performance

# 4. 메모리 효율적인 작은 배치:
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_memory_efficient_predictions.txt \
#     --output_jsonl_file ./t5_memory_efficient_predictions.json \
#     --batch_size 4 \
#     --measure_performance

# 5. 빠른 처리용 큰 배치:
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_fast_predictions.txt \
#     --output_jsonl_file ./t5_fast_predictions.json \
#     --batch_size 32 \
#     --measure_performance

# 6. 다른 pretrained 모델 사용:
# CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --pretrained_model t5-small \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./t5_small_batch_predictions.txt \
#     --output_jsonl_file ./t5_small_batch_predictions.json \
#     --batch_size 16 \
#     --measure_performance

# T5 학습 데이터 생성:
# python t5_slot_filling_batch.py \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/train_with_literals.jsonl \
#     --create_training_data \
#     --training_data_output ./t5_training_data.jsonl 


# export CUDA_VISIBLE_DEVICES=0 python t5_slot_filling_batch.py \
#     --model_path ./t5_slot_filling_model \
#     --input_file /data/yhyunjun/T2C/dataset/cypherbench/Pre-SF/sfr-sfr/test_with_literals.jsonl \
#     --output_txt_file ./trained_model_latency_predictions.txt \
#     --output_jsonl_file ./trained_model_latency_predictions.json \
#     --num_examples 10 \
#     --batch_size 1 \
#     --measure_performance