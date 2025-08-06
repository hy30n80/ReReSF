import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from eval import evaluate_ranking_metrics
from collections import defaultdict
import pdb as pdb
from datetime import timedelta
import numpy as np
from tabulate import tabulate
import torch
import csv
import os
from tqdm import tqdm
import sys



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'


def encode_passages_batched(passages, p_encoder, p_tokenizer, device, batch_size=128):
    all_reps = []
    p_encoder.eval()

    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
        batch = passages[i:i + batch_size]
        inputs = p_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = p_encoder(**inputs)
            reps = outputs.last_hidden_state[:, 0]  # ⬅️ [CLS] token embedding

        all_reps.append(reps.cpu())

    return torch.cat(all_reps, dim=0).to(device)


class TextSQLDataset(Dataset):
    def __init__(self, data_path, q_tokenizer, p_tokenizer):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # JSON 배열 형식 처리
            self.samples = json.load(f)
        self.q_tokenizer = q_tokenizer
        self.p_tokenizer = p_tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['nl_question']
        positive = sample['masked_cypher']

        q_inputs = self.q_tokenizer(question, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        p_inputs = self.p_tokenizer(positive, return_tensors='pt', truncation=True, padding='max_length', max_length=256)

        return {
            'q_input_ids': q_inputs['input_ids'].squeeze(0),
            'q_attention_mask': q_inputs['attention_mask'].squeeze(0),
            'p_input_ids': p_inputs['input_ids'].squeeze(0),  # <-- 이 줄 추가!
            'p_attention_mask': p_inputs['attention_mask'].squeeze(0)  # <-- 이것도!
        }



def load_test_dataset(train_json_path, test_json_path, all_json_path):

    # 1. Load test and all_data (JSON 배열 형식 처리)
    with open(train_json_path, encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_json_path, encoding="utf-8") as f:
        test_data = json.load(f)

    with open(all_json_path, encoding="utf-8") as f:
        all_data = json.load(f)


    # 2. Group test_data and all_data by graph
    train_by_graph = defaultdict(list)
    for ex in train_data:
        train_by_graph[ex['graph']].append(ex)

    test_by_graph = defaultdict(list)
    for ex in test_data:
        test_by_graph[ex['graph']].append(ex)

    all_by_graph = defaultdict(list)
    for ex in all_data:
        if 'graph' in ex:
            all_by_graph[ex['graph']].append(ex)

    
    return train_by_graph, test_by_graph, all_by_graph





def generate_cross_encoder_data(split_name, target_by_graph, all_by_graph,
                                q_encoder, p_encoder, q_tokenizer, p_tokenizer,
                                device, save_path, top_k=20):
    q_encoder.eval()
    p_encoder.eval()

    output = []
    total_skipped = 0

    for graph in list(target_by_graph.keys()):
        test_data = target_by_graph[graph]
        all_data = all_by_graph[graph]
        queries = [ex['nl_question'] for ex in test_data]
        passages = [ex['masked_cypher'] for ex in all_data]

        # Encode all passages for this graph
        p_reps = encode_passages_batched(passages, p_encoder, p_tokenizer, device, batch_size=128)

        for i, q in enumerate(tqdm(queries, desc=f"[{split_name}] Graph: {graph}")):
            gold_cypher = test_data[i]['masked_cypher']

            q_inputs = q_tokenizer(q, return_tensors='pt', padding=True, truncation=True, max_length=128)
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            q_outputs = q_encoder(**q_inputs)
            q_rep = q_outputs.last_hidden_state[:, 0]  # [CLS] embedding

            scores = torch.matmul(q_rep, p_reps.T).squeeze(0)  # (N,)
            sorted_indices = torch.argsort(scores, descending=True)
            top_indices = sorted_indices[:top_k].tolist()
            top_passages = [passages[idx] for idx in top_indices]

            if gold_cypher not in top_passages:
                total_skipped += 1
                continue  # skip examples where gold is not in top-K

            label = top_passages.index(gold_cypher)

            # test_data[i]에 있는 원본 딕셔너리를 그대로 가져옵니다.
            new_entry = test_data[i]

            # 여기에 'candidates'와 'label'이라는 새로운 키와 값을 추가합니다.
            new_entry['candidates'] = top_passages
            new_entry['label'] = label

            # 기존 정보가 모두 유지된 상태의 딕셔너리를 output에 추가합니다.
            output.append(new_entry)

    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for ex in output:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[✔] Saved {len(output)} examples to {save_path}")
    print(f"[!] Skipped {total_skipped} examples (gold not in top-{top_k})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path", default="../datasets/cypherbench")
    parser.add_argument("-r", "--result_path", default="./results")
    parser.add_argument("-o", "--output_path", default="../outputs/cypherbench/retrieval_results")

    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-m", "--model_path", default="./models/SFR") # Trained Retriever Model Path
    args = parser.parse_args()


    # Dataset 경로
    dataset_path = args.dataset_path
    train_path = f"{dataset_path}/train.json"
    test_path = f"{dataset_path}/test.json"
    all_path = f"{dataset_path}/augmented_cypher.json"

    # Retrieval 결과 저장할 경로
    output_path = args.output_path
    saved_train_path = f"{output_path}/train.json"
    saved_test_path = f"{output_path}/test.json"

    # Retrieval Accuracy 저장 
    result_path = args.result_path
    save_path_seen = f"{result_path}/train_ep.csv"
    save_path_unseen = f"{result_path}/test_ep.csv"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    model_name = "Salesforce/SFR-Embedding-Code-400M_R"
    # 토크나이저 및 모델 로드
    q_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    p_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_path = args.model_path 

    q_model_path = os.path.join(model_path, "query_enc_model")
    p_model_path = os.path.join(model_path, "sql_enc_model")
    q_encoder = AutoModel.from_pretrained(q_model_path, trust_remote_code=True).to(device)
    p_encoder = AutoModel.from_pretrained(p_model_path, trust_remote_code=True).to(device)


    train_by_graph, test_by_graph, all_by_graph = load_test_dataset(train_path, test_path, all_path)


    metrics = evaluate_ranking_metrics(
        q_encoder, p_encoder, train_by_graph, test_by_graph, all_by_graph,
        q_tokenizer, p_tokenizer, device="cuda", 
        save_path_seen = save_path_seen, 
        save_path_unseen = save_path_unseen, 
        epoch=0
    
    )

    # Generate Cross-Encoder fine-tuning data
    generate_cross_encoder_data("train", train_by_graph, all_by_graph, q_encoder, p_encoder,
                                q_tokenizer, p_tokenizer, device,
                                save_path=saved_train_path)

    generate_cross_encoder_data("test", test_by_graph, all_by_graph, q_encoder, p_encoder,
                                q_tokenizer, p_tokenizer, device,
                                save_path=saved_test_path)