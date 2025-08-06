# in-batch negative sampling - 아니야 얘 

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



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'

parser = argparse.ArgumentParser()


parser.add_argument("-i", "--train_file_path", default="../datasets/cypherbench/train.json")
parser.add_argument("-o", "--model_path", default="./models/SFR")
parser.add_argument("-ep", "--epochs", type=int, default=21)
parser.add_argument("-bs", "--batch_size", type=int, default=32) 
parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5)
parser.add_argument("-r", "--result_path", default="SFR")


args = parser.parse_args()

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




def dpr_loss(q_reps, p_reps):
    scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
    labels = torch.arange(scores.size(0)).to(scores.device)
    return torch.nn.functional.cross_entropy(scores, labels)



def load_test_dataset():
    # 파일 경로
    dataset_path = "../datasets/cypherbench"
    train_json_path = f"{dataset_path}/train.json"
    test_json_path = f"{dataset_path}/test.json"
    all_json_path = f"{dataset_path}/augmented_cypher.json"

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
        if ex['graph'] in ["soccer", "terrorist_attack", "art", "biology"]:
            train_by_graph[ex['graph']].append(ex)

    test_by_graph = defaultdict(list)
    for ex in test_data:
        test_by_graph[ex['graph']].append(ex)

    all_by_graph = defaultdict(list)
    for ex in all_data:
        if 'graph' in ex:
            all_by_graph[ex['graph']].append(ex)
    
    return train_by_graph, test_by_graph, all_by_graph




def train(rank, world_size, data_path):

    print("-------------Train Mode-------------")

    # 프로세스 그룹 초기화
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

    # Test-set, All
    train_by_graph, test_by_graph, all_by_graph = load_test_dataset()

    if rank == 0:
        log_dirpath = os.path.join(args.out_dirpath, "logs")
        writer = SummaryWriter(log_dir=log_dirpath)
    
    model_name = "Salesforce/SFR-Embedding-Code-400M_R"

    # 토크나이저 및 모델 로드
    q_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    p_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    q_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(rank)
    p_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(rank)

    # DDP로 모델 래핑
    q_encoder = DDP(q_encoder, device_ids=[rank])
    p_encoder = DDP(p_encoder, device_ids=[rank])

    # 데이터셋 및 데이터로더 설정
    dataset = TextSQLDataset(data_path, q_tokenizer, p_tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=int(args.batch_size), sampler=sampler)

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(list(q_encoder.parameters()) + list(p_encoder.parameters()), lr=float(args.learning_rate))

    # 학습 루프
    for epoch in range(int(args.epochs)):
        sampler.set_epoch(epoch)
        
        if (epoch == 1 or epoch % 5 == 0) and rank == 0:
            # metric 계산
            metrics = evaluate_ranking_metrics(
                q_encoder.module, p_encoder.module, train_by_graph, test_by_graph, all_by_graph,
                q_tokenizer, p_tokenizer, device=rank, 
                save_path_seen = f"./results/{args.result_path}_seen.csv", 
                save_path_unseen = f"./results/{args.result_path}_unseen.csv", 
                epoch=epoch
            )
            writer.add_scalar("Eval/Recall@1", metrics["recall@1"], epoch)
            writer.add_scalar("Eval/Recall@5", metrics["recall@5"], epoch)
            writer.add_scalar("Eval/Recall@10", metrics["recall@10"], epoch)
            writer.add_scalar("Eval/Recall@20", metrics["recall@20"], epoch)
            writer.add_scalar("Eval/MRR", metrics["mrr"], epoch)

            out_encoder_path = os.path.join(args.out_dirpath)
            q_encoder.module.save_pretrained(os.path.join(out_encoder_path, "query_enc_model"))
            p_encoder.module.save_pretrained(os.path.join(out_encoder_path, "sql_enc_model"))

        dist.barrier()

        for step_idx, batch in enumerate(dataloader):
            q_input_ids = batch['q_input_ids'].to(rank)
            q_attention_mask = batch['q_attention_mask'].to(rank)
            p_input_ids = batch['p_input_ids'].to(rank)
            p_attention_mask = batch['p_attention_mask'].to(rank)

            q_outputs = q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
            p_outputs = p_encoder(input_ids=p_input_ids, attention_mask=p_attention_mask)


            # SFR 공식 방식: [CLS] 토큰 임베딩 사용
            q_reps = q_outputs.last_hidden_state[:, 0]  # (B, D)
            p_reps = p_outputs.last_hidden_state[:, 0]  # (B * num_pos, D)

            loss = dpr_loss(q_reps, p_reps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                global_step = epoch * len(dataloader) + step_idx
                writer.add_scalar("Loss/train", loss.item(), global_step)
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    
    if rank == 0:
        writer.close()
    # 프로세스 그룹 종료
    dist.destroy_process_group()


def main():
    data_path = args.train_file_path
    world_size = torch.cuda.device_count()
    print("# GPUs: ", world_size)
    mp.spawn(train, args=(world_size, data_path), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
