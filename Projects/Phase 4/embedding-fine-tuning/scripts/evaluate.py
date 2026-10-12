"""Evaluation Script for Information Retrieval Benchmark.

Evaluates SentenceTransformer models on the fixed PyTorch documentation retrieval corpus.
Enforces benchmark integrity:
- Preserves test split freeze (only validation split evaluated unless explicit override)
- Never modifies dataset files
- Saves evaluation results safely without overwriting existing files
- Records nDCG@10, MRR@10, Recall@1, Recall@5, Accuracy@k, and MAP@100
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

# Set offline mode to avoid unnecessary HuggingFace Hub network retries
os.environ["HF_HUB_OFFLINE"] = "1"

# Directories
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def load_benchmark_data(split: str = "val", allow_test: bool = False):
    """Load evaluation corpus, queries, and relevance judgments.
    
    Strictly safeguards test set from being loaded accidentally during pre-training.
    """
    if split == "test" and not allow_test:
        raise PermissionError(
            "Evaluation on the TEST split is locked until post-fine-tuning evaluation! "
            "Pass --allow-test only after fine-tuning is completed."
        )

    corpus_path = DATA_DIR / "corpus.json"
    queries_path = DATA_DIR / f"{split}_queries.json"
    qrels_path = DATA_DIR / f"{split}_qrels.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    # Read data strictly in read-only mode
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data: dict[str, dict[str, Any]] = json.load(f)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data: dict[str, str] = json.load(f)

    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_data: dict[str, dict[str, int]] = json.load(f)

    # Format for InformationRetrievalEvaluator
    corpus = {doc_id: doc["text"] for doc_id, doc in corpus_data.items()}
    queries = {qid: text for qid, text in queries_data.items()}
    relevant_docs = {qid: set(rel_docs.keys()) for qid, rel_docs in qrels_data.items()}

    return corpus, queries, relevant_docs, corpus_data, queries_data, qrels_data


def run_evaluation(
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    split: str = "val",
    allow_test: bool = False,
    output_filename: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute information retrieval evaluation and record metrics."""
    print("=" * 70)
    print("INFORMATION RETRIEVAL EVALUATION BENCHMARK")
    print("=" * 70)
    print(f"Model:      {model_name_or_path}")
    print(f"Split:      {split}")
    print(f"Timestamp:  {datetime.datetime.now().isoformat()}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    if device == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # 1. Load benchmark data
    corpus, queries, relevant_docs, corpus_raw, queries_raw, qrels_raw = load_benchmark_data(
        split=split, allow_test=allow_test
    )
    total_judgments = sum(len(docs) for docs in relevant_docs.values())
    print(f"Corpus:     {len(corpus)} passages (search space)")
    print(f"Queries:    {len(queries)} queries ({total_judgments} total relevant judgments)")

    # 2. Load model
    print(f"\nLoading model '{model_name_or_path}'...")
    model = SentenceTransformer(model_name_or_path, device=device)
    model.eval()

    # 3. Setup InformationRetrievalEvaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=split,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[100],
        show_progress_bar=False,
        batch_size=batch_size,
        write_csv=False,
    )

    # 4. Run evaluation
    print(f"Running evaluation over {len(queries)} queries and {len(corpus)} corpus passages...")
    eval_results = evaluator(model)

    # Extract primary required metrics
    prefix = f"{split}_cosine_"
    ndcg_10 = float(eval_results.get(f"{prefix}ndcg@10", 0.0))
    mrr_10 = float(eval_results.get(f"{prefix}mrr@10", 0.0))
    recall_1 = float(eval_results.get(f"{prefix}recall@1", 0.0))
    recall_5 = float(eval_results.get(f"{prefix}recall@5", 0.0))
    recall_10 = float(eval_results.get(f"{prefix}recall@10", 0.0))
    acc_1 = float(eval_results.get(f"{prefix}accuracy@1", 0.0))
    acc_5 = float(eval_results.get(f"{prefix}accuracy@5", 0.0))
    acc_10 = float(eval_results.get(f"{prefix}accuracy@10", 0.0))
    map_100 = float(eval_results.get(f"{prefix}map@100", 0.0))

    primary_metrics = {
        "nDCG@10": ndcg_10,
        "MRR@10": mrr_10,
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "Recall@10": recall_10,
        "Accuracy@1": acc_1,
        "Accuracy@5": acc_5,
        "Accuracy@10": acc_10,
        "MAP@100": map_100,
    }

    # 5. Compute detailed per-query rankings for full transparency
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    
    with torch.no_grad():
        q_emb = model.encode(list(queries.values()), convert_to_tensor=True, device=device)
        c_emb = model.encode(list(corpus.values()), convert_to_tensor=True, device=device)
        sim_matrix = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(1), c_emb.unsqueeze(0), dim=-1
        )

    per_query_details = []
    for i, qid in enumerate(query_ids):
        scores, indices = torch.topk(sim_matrix[i], k=10)
        gold_docs = relevant_docs[qid]
        top10_hits = []
        ranks_of_gold = []
        for rank, (score_val, doc_idx) in enumerate(zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1):
            did = corpus_ids[doc_idx]
            is_relevant = did in gold_docs
            if is_relevant:
                ranks_of_gold.append(rank)
            top10_hits.append({
                "rank": rank,
                "doc_id": did,
                "score": round(score_val, 4),
                "title": corpus_raw[did]["title"],
                "is_relevant": is_relevant,
            })
        
        per_query_details.append({
            "query_id": qid,
            "query": queries[qid],
            "gold_doc_ids": list(gold_docs),
            "hits_in_top10": [h["doc_id"] for h in top10_hits if h["is_relevant"]],
            "ranks_of_gold_in_top10": ranks_of_gold,
            "first_relevant_rank": ranks_of_gold[0] if ranks_of_gold else None,
            "top3_retrieved": top10_hits[:3],
        })

    # 6. Format full results artifact
    full_output = {
        "evaluation_name": f"{Path(model_name_or_path).stem}_{split}_evaluation",
        "model_name_or_path": model_name_or_path,
        "split": split,
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "benchmark_statistics": {
            "corpus_size": len(corpus),
            "queries_count": len(queries),
            "total_relevant_judgments": total_judgments,
        },
        "primary_metrics": primary_metrics,
        "raw_evaluator_metrics": {k: float(v) for k, v in eval_results.items()},
        "per_query_results": per_query_details,
    }

    # 7. Safe save without overwriting
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        target_file = RESULTS_DIR / f"{model_name_or_path.replace('/', '_')}_{split}_metrics.json"
    else:
        target_file = RESULTS_DIR / output_filename

    if target_file.exists():
        timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_file = target_file.with_name(f"{target_file.stem}_{timestamp_suffix}{target_file.suffix}")
        print(f"Target file already existed. Saved to timestamped file to prevent overwrite: {target_file}")
    else:
        print(f"Saving baseline results to: {target_file}")

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"nDCG@10:    {ndcg_10:.4f}  ({ndcg_10 * 100:.2f}%)")
    print(f"MRR@10:     {mrr_10:.4f}  ({mrr_10 * 100:.2f}%)")
    print(f"Recall@1:   {recall_1:.4f}  ({recall_1 * 100:.2f}%)")
    print(f"Recall@5:   {recall_5:.4f}  ({recall_5 * 100:.2f}%)")
    print(f"Recall@10:  {recall_10:.4f}  ({recall_10 * 100:.2f}%)")
    print(f"Accuracy@1: {acc_1:.4f}  ({acc_1 * 100:.2f}%)")
    print(f"Accuracy@5: {acc_5:.4f}  ({acc_5 * 100:.2f}%)")
    print(f"MAP@100:    {map_100:.4f}  ({map_100 * 100:.2f}%)")
    print(f"Results file saved at: {target_file}")
    print("=" * 70)

    return full_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate SentenceTransformer on retrieval benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name on HF Hub or local checkpoint path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split (default: val)",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly permit evaluation on test split (prohibited for baseline)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="baseline_val_metrics.json",
        help="Output filename within results directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size",
    )

    args = parser.parse_args()

    run_evaluation(
        model_name_or_path=args.model,
        split=args.split,
        allow_test=args.allow_test,
        output_filename=args.output_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
