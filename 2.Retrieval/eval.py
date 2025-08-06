import numpy as np
from tabulate import tabulate
import torch
import csv
import os
from tqdm import tqdm
from collections import defaultdict



def encode_passages_batched(passages, p_encoder, p_tokenizer, device, batch_size=32):
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



def evaluate_ranking_metrics(q_encoder, p_encoder, train_by_graph, test_by_graph, all_by_graph,
        q_tokenizer, p_tokenizer, device, epoch, top_k=20, save_path_seen="./results/seen.csv", save_path_unseen="./results/unseen.csv"):
    q_encoder.eval()
    p_encoder.eval()

    with torch.no_grad():
        for mode in ["train", "test"]:
            if mode == "train":
                print("Seen DB, Seen Gold Pairs! \n")
                target = train_by_graph
            else:
                print("Unseen DB, Unseen Gold Pairs! \n")
                target = test_by_graph

            total_size = 0
            weighted_sums = defaultdict(float)
            overall_ranks = []

            for graph in list(target.keys()):
                test_data = target[graph]
                all_data = all_by_graph[graph]
                queries = [ex['nl_question'] for ex in test_data]
                passages = [ex['masked_cypher'] for ex in all_data]

                p_reps = encode_passages_batched(passages, p_encoder, p_tokenizer, device, batch_size=128)

                ranks = []
                losses = []

                for i, q in enumerate(queries):
                    q_inputs = q_tokenizer(q, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
                    q_outputs = q_encoder(**q_inputs)
                    q_rep = q_outputs.last_hidden_state[:, 0]  # ⬅️ [CLS] token embedding

                    scores = torch.matmul(q_rep, p_reps.T).squeeze(0)
                    sorted_indices = torch.argsort(scores, descending=True)

                    gold_cypher = test_data[i]['masked_cypher']
                    true_indices = [j for j, p in enumerate(passages) if p == gold_cypher]

                    ranks_of_true = [rank + 1 for rank, idx in enumerate(sorted_indices.tolist()) if idx in true_indices]
                    rank = min(ranks_of_true) if ranks_of_true else len(passages) + 1
                    ranks.append(rank)

                    if true_indices:
                        label = torch.tensor([true_indices[0]], dtype=torch.long, device=device)
                        loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), label)
                        losses.append(loss.item())

                ranks = np.array(ranks)
                size = len(test_data)
                total_size += size

                recall_at = lambda k: np.mean(ranks <= k)
                mrr = np.mean(1 / ranks)
                mean_rank = int(np.mean(ranks))
                max_rank = int(np.max(ranks))
                avg_test_loss = np.mean(losses)

                for k, v in {
                    "recall@1": recall_at(1),
                    "recall@5": recall_at(5),
                    "recall@10": recall_at(10),
                    "recall@20": recall_at(20),
                    "mrr": mrr,
                    "avg_test_loss": avg_test_loss
                }.items():
                    weighted_sums[k] += v * size

                overall_ranks.extend(ranks.tolist())

            final_metrics = {
                k: weighted_sums[k] / total_size
                for k in weighted_sums
            }
            all_ranks = np.array(overall_ranks)
            final_metrics["mean_rank"] = int(np.mean(all_ranks))
            final_metrics["max_rank"] = int(np.max(all_ranks))

            save_path = save_path_seen if mode == "train" else save_path_unseen

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file_exists = os.path.exists(save_path)

                with open(save_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Epoch", "Recall@1", "Recall@5", "Recall@10", "Recall@20",
                            "MRR", "Mean Rank", "Max Rank", "Avg Test Loss"
                        ])

                    writer.writerow([
                        epoch if epoch is not None else "-",
                        f"{final_metrics['recall@1']:.4f}",
                        f"{final_metrics['recall@5']:.4f}",
                        f"{final_metrics['recall@10']:.4f}",
                        f"{final_metrics['recall@20']:.4f}",
                        f"{final_metrics['mrr']:.4f}",
                        f"{final_metrics['mean_rank']:.2f}",
                        final_metrics["max_rank"],
                        f"{final_metrics['avg_test_loss']:.4f}"
                    ])

                print(f"[\u2714] Final metrics saved to: {save_path}")

            result_table = [
                ["Recall@1", f"{final_metrics['recall@1']:.4f}"],
                ["Recall@5", f"{final_metrics['recall@5']:.4f}"],
                ["Recall@10", f"{final_metrics['recall@10']:.4f}"],
                ["Recall@20", f"{final_metrics['recall@20']:.4f}"],
                ["MRR", f"{final_metrics['mrr']:.4f}"],
                ["Mean Rank", f"{final_metrics['mean_rank']:.4f}"],
                ["Max Rank", f"{final_metrics['max_rank']:.4f}"],
                ["Avg Test Loss", f"{final_metrics['avg_test_loss']:.4f}"]
            ]

            print("\n" + tabulate(result_table, headers=["Metric", "Value"], tablefmt="pretty") + "\n")

    q_encoder.train()
    p_encoder.train()
    return final_metrics
