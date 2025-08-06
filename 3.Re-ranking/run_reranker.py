import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel

# 1. SFR 모델을 위한 커스텀 분류 클래스 정의
class SFRForSequenceClassification(nn.Module):
    def __init__(self, model_name_or_path, num_labels=1):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
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

    @classmethod
    def from_pretrained(cls, model_path, num_labels=1):
        """커스텀 모델과 학습된 가중치를 불러오는 클래스 메소드"""
        model = cls(model_path, num_labels)
        classifier_path = os.path.join(model_path, "classifier.pt")
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        else:
            print(f"Warning: 'classifier.pt' not found in {model_path}. Classifier weights are not loaded.")
        return model


class ReRankDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["nl_question"]
        # candidates가 리스트의 리스트일 경우를 대비하여 처리
        candidates = [c[0] if isinstance(c, list) else c for c in sample["candidates"]]
        
        encoded = self.tokenizer(
            [question]*len(candidates),
            candidates,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

@torch.no_grad()
def rerank(model, tokenizer, data_path, save_path, device="cuda"):
    model.eval()
    model.to(device)

    # 2. 데이터 보존을 위해 원본 데이터를 미리 로드
    with open(data_path, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f]

    dataset = ReRankDataset(original_data, tokenizer)
    # Reranking은 1개 샘플씩 처리하므로 batch_size=1
    dataloader = DataLoader(dataset, batch_size=1)

    with open(save_path, 'w', encoding='utf-8') as fw:
        # 원본 데이터와 dataloader를 함께 순회
        for original_sample, batch in zip(original_data, tqdm(dataloader, desc=f"Reranking {os.path.basename(data_path)}")):
            input_ids = batch['input_ids'].squeeze(0).to(device)       # (K, L)
            attention_mask = batch['attention_mask'].squeeze(0).to(device)
            
            # 원본 데이터에서 정보 가져오기
            candidates = original_sample["candidates"]
            # 후보 리스트가 이중 리스트로 되어 있을 경우를 대비
            candidates = [c[0] if isinstance(c, list) else c for c in candidates]
            gt_cypher = candidates[original_sample["label"]]

            # Re-ranking 실행
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.squeeze(-1) # 모델이 (K, 1)을 반환하므로 차원 축소
            sorted_indices = torch.argsort(scores, descending=True)
            reranked_candidates = [candidates[i] for i in sorted_indices.tolist()]

            # Re-rank된 리스트에서 새로운 정답 위치 찾기
            try:
                new_label = reranked_candidates.index(gt_cypher)
            except ValueError:
                new_label = -1 # Re-rank 후 Top-K 안에 정답이 없는 경우

            # 3. 원본 데이터를 복사하고, 필요한 필드만 업데이트
            output_entry = original_sample.copy()
            output_entry["candidates"] = reranked_candidates
            output_entry["label"] = new_label
            
            fw.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

def process_folder(model, tokenizer, input_folder, output_folder, device="cuda"):
    """폴더 내의 train.json과 test.json을 한 번에 처리"""
    print(f"Processing folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 처리할 파일들
    files_to_process = ["train.json", "test.json"]
    
    for filename in files_to_process:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if os.path.exists(input_path):
            print(f"\nProcessing {filename}...")
            rerank(model, tokenizer, input_path, output_path, device)
            print(f"✓ Completed: {filename}")
        else:
            print(f"⚠ Warning: {input_path} not found, skipping...")
    
    print(f"\n[✔] All files processed. Results saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker_model_dir", type=str, required=True, help="Path to fine-tuned re-ranker model (e.g., models/SFR)")
    parser.add_argument("--retrieval_results_folder", type=str, required=True, help="Path to input folder containing train.json and test.json")
    parser.add_argument("--reranking_results_folder", type=str, required=True, help="Path to output folder for reranked results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. 모델 로딩 로직 변경    
    print(f"Loading tokenizer from: {args.reranker_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_dir)
    
    print(f"Loading SFR model from: {args.reranker_model_dir}")
    # 커스텀 클래스의 from_pretrained 메소드를 사용하여 모델 로드
    model = SFRForSequenceClassification.from_pretrained(args.reranker_model_dir, num_labels=1)

    process_folder(model, tokenizer, args.retrieval_results_folder, args.reranking_results_folder, device=device)
    
    print(f"\n[✔] Reranking complete. Saved to: {args.reranking_results_folder}")








