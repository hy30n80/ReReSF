import os
import json
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from transformers import AutoModel

# DDP 설정
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12356"

# ==================== 개선 1: ArgumentParser 정의만 남겨둠 ====================
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--reranker_model_name", type=str, default="SFR")
parser.add_argument("-i", "--retrieval_results_path", type=str, default="../outputs/cypherbench/retrieval_results/SFR") # Retreival (SFR) 결과를 기반으로 Re-ranker 학습 -> Retreiver 결과 파일
parser.add_argument("-o", "--model_path", type=str, default="./models/SFR")
parser.add_argument("-ep", "--epochs", type=int, default=21) #21
parser.add_argument("-bs", "--batch_size", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5)
parser.add_argument("-r", "--result_path", type=str, default="./results/SFR")



@torch.no_grad()
def evaluate_reranker_multi_recall(model, tokenizer, data_path, rank, world_size, batch_size=32, ks=[1, 5, 10, 20]):
    model.eval()
    dataset = CrossEncoderReRankDataset(data_path, tokenizer)
    # ==================== 개선 1: 평가용 Sampler 추가 ====================
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 결과를 각 GPU에 텐서로 저장
    recall_hits_tensor = torch.zeros(len(ks), dtype=torch.long).to(rank)
    total_tensor = torch.zeros(1, dtype=torch.long).to(rank)

    loop = tqdm(dataloader, desc=f"[Eval] {os.path.basename(data_path)}", disable=(rank != 0))
    for batch in loop:
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['label'].to(rank)

        B, K, L = input_ids.shape
        input_ids = input_ids.view(B * K, L)
        attention_mask = attention_mask.view(B * K, L)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.view(B, K)

        topk = torch.topk(logits, max(ks), dim=1).indices
        for i, k in enumerate(ks):
            hits = (topk[:, :k] == labels.unsqueeze(1)).any(dim=1).sum()
            recall_hits_tensor[i] += hits

        total_tensor[0] += B

    # ==================== 개선 2: 모든 GPU의 결과를 합산 (Reduce) ====================
    dist.all_reduce(recall_hits_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    model.train()

    if rank == 0:
        # 0번 GPU에서만 최종 점수 계산 및 반환
        total = total_tensor.item()
        recall_hits = recall_hits_tensor.tolist()
        recall_scores = {f"recall@{k}": recall_hits[i] / total for i, k in enumerate(ks)}
        return recall_scores
    else:
        # 다른 GPU들은 None 반환
        return None




def save_multi_recall_csv(epoch, train_metrics, test_metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.exists(save_path)

    ks = [int(k.replace("recall@", "")) for k in train_metrics.keys()]
    header = ["Epoch"] + [f"Train_R@{k}" for k in ks] + [f"Test_R@{k}" for k in ks]
    row = [epoch] + [f"{train_metrics[f'recall@{k}']:.4f}" for k in ks] + \
                  [f"{test_metrics[f'recall@{k}']:.4f}" for k in ks]

    with open(save_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


class CrossEncoderReRankDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        self.samples = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8')]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['nl_question']
        candidates = sample['candidates']
        label = sample['label']
        encoded = self.tokenizer([question]*len(candidates), candidates, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }


def crossentropy_rerank_loss(logits, label_idx):
    return torch.nn.functional.cross_entropy(logits, label_idx)

class SFRForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        config = self.base_model.config
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits
    
    # ==================== 개선 2: 모델 저장을 위한 save_pretrained 메소드 추가 ====================
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Base 모델 저장
        self.base_model.save_pretrained(save_directory)
        # Classifier의 state_dict 저장
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))


def train(rank, world_size, args, train_path, test_path):
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

    model_name = "Salesforce/SFR-Embedding-Code-400M_R"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = SFRForSequenceClassification(model_name=model_name, num_labels=1).to(rank)
    model = DDP(model, device_ids=[rank])

    dataset = CrossEncoderReRankDataset(train_path, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.model_path, "logs"))

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        # ==================== 개선 3: 평가 호출을 모든 프로세스가 하도록 변경 ====================
        # if epoch == 1 or epoch % 5 == 0:
        if True:
            eval_batch_size = 64

            # 모든 프로세스가 평가 함수를 호출
            train_metrics = evaluate_reranker_multi_recall(model.module, tokenizer, train_path, rank, world_size, batch_size=eval_batch_size)
            test_metrics = evaluate_reranker_multi_recall(model.module, tokenizer, test_path, rank, world_size, batch_size=eval_batch_size)

            # 결과 처리 및 저장은 0번 프로세스만 담당
            if rank == 0:
                print(f"[Eval @ Epoch {epoch}]")
                for k in train_metrics:
                    print(f"Train {k}: {train_metrics[k]:.4f} | Test {k}: {test_metrics[k]:.4f}")

                save_multi_recall_csv(epoch, train_metrics, test_metrics, os.path.join(args.result_path, "reranker_results.csv"))

                out_path = os.path.join(args.model_path)
                model.module.save_pretrained(out_path)
                tokenizer.save_pretrained(out_path)

        # 모든 프로세스가 여기서 동기화될 때까지 기다림
        dist.barrier()

        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Epoch {epoch}] Rank {rank}")
        for step, batch in loop:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['label'].to(rank)

            B, K, L = input_ids.size()
            input_ids = input_ids.view(B * K, L)
            attention_mask = attention_mask.view(B * K, L)

            optimizer.zero_grad()
            
            # ==================== 개선 4: AMP(autocast) 적용 ====================
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.view(B, K)
                loss = crossentropy_rerank_loss(logits, labels)
            
            # scaler를 사용해 loss를 스케일링하고 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # =================================================================

            loop.set_postfix(loss=f"{loss.item():.4f}")
            if rank == 0:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar("Loss/train", loss.item(), global_step)


    if rank == 0:
        writer.close()

    dist.destroy_process_group()

def main():
    # ==================== 개선 3: 파싱 및 경로 설정을 main 함수 내부로 이동 ====================
    args = parser.parse_args()

    print("====Training Re-ranker====")
    train_path = f"{args.retrieval_results_path}/train.json" # Retreiver (SFR) 을 통해, Train-set NLQ 의 Top-20 추출한 파일
    test_path = f"{args.retrieval_results_path}/test.json" # Retreiver (SFR) 을 통해, Test-set NLQ 의 Top-20 추출한 파일

    args.result_path = f"results/{args.reranker_model_name}" # Re-ranker 결과 파일
    args.model_path = f"models/{args.reranker_model_name}" # Re-ranker 모델 파일

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, train_path, test_path), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()