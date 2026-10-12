"""Fine-Tuning Script for Sentence Transformers with Contrastive Learning (Phase 3).

Trains sentence-transformers/all-MiniLM-L6-v2 on 150 technical PyTorch documentation pairs
using MultipleNegativesRankingLoss (MNRL) and evaluates against the validation benchmark.

Enforces benchmark integrity:
- Test split (test_queries.json, test_qrels.json) is strictly untouched.
- Dataset files remain read-only and unmodified.
- Saves fine-tuned model to a dedicated new directory without overwriting base models.
- Saves post-training validation metrics to results/finetuned_val_metrics.json without overwriting baseline results.
- Optimized for RTX 3050 Laptop GPU (4 GB VRAM) using BF16 mixed precision and TF32.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from transformers import get_linear_schedule_with_warmup

# Set offline mode to avoid unnecessary HuggingFace Hub network retries
os.environ["HF_HUB_OFFLINE"] = "1"

# Directories
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_for_training():
    """Load train pairs and validation benchmark data.
    
    Strictly safeguards test split: test_queries.json and test_qrels.json are never accessed.
    """
    train_path = DATA_DIR / "train_pairs.json"
    corpus_path = DATA_DIR / "corpus.json"
    val_queries_path = DATA_DIR / "val_queries.json"
    val_qrels_path = DATA_DIR / "val_qrels.json"

    for p in [train_path, corpus_path, val_queries_path, val_qrels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required benchmark file not found: {p}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_pairs = json.load(f)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    with open(val_queries_path, "r", encoding="utf-8") as f:
        val_queries = json.load(f)

    with open(val_qrels_path, "r", encoding="utf-8") as f:
        val_qrels_raw = json.load(f)

    corpus = {doc_id: doc["text"] for doc_id, doc in corpus_data.items()}
    val_relevant_docs = {qid: set(rel_docs.keys()) for qid, rel_docs in val_qrels_raw.items()}

    return train_pairs, corpus, val_queries, val_relevant_docs, corpus_data, val_qrels_raw


def train_model(
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    scale: float = 20.0,
    seed: int = 42,
    output_model_dir: Path | None = None,
    results_file: Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        gpu_name = "CPU"

    print("=" * 75)
    print("PHASE 3: CONTRASTIVE FINE-TUNING FOR SENTENCE TRANSFORMERS")
    print("=" * 75)
    print(f"Base Model:       {model_name_or_path}")
    print(f"Device:           {device} ({gpu_name})")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print(f"Weight Decay:     {weight_decay}")
    print(f"MNRL Scale:       {scale} (Temperature = {1.0 / scale:.4f})")
    print(f"Seed:             {seed}")
    print(f"Timestamp:        {datetime.datetime.now().isoformat()}")

    # 1. Load Data
    train_pairs, corpus, val_queries, val_relevant_docs, corpus_raw, val_qrels_raw = load_dataset_for_training()
    print(f"\nLoaded {len(train_pairs)} training query-passage pairs.")
    print(f"Validation search space: {len(corpus)} passages, {len(val_queries)} queries.")

    # 2. Initialize Model & Loss
    print(f"\nLoading base model '{model_name_or_path}'...")
    model = SentenceTransformer(model_name_or_path, local_files_only=True, device=device)
    loss_fct = MultipleNegativesRankingLoss(model, scale=scale)

    # 3. Setup Validation Evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=corpus,
        relevant_docs=val_relevant_docs,
        name="val",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[100],
        show_progress_bar=False,
        batch_size=32,
        write_csv=False,
    )

    # Pre-training baseline evaluation
    print("\nEvaluating initial baseline performance on validation benchmark...")
    model.eval()
    baseline_metrics = evaluator(model)
    prefix = "val_cosine_"
    print(f"  Baseline nDCG@10:  {baseline_metrics.get(f'{prefix}ndcg@10', 0):.4f}")
    print(f"  Baseline MRR@10:   {baseline_metrics.get(f'{prefix}mrr@10', 0):.4f}")
    print(f"  Baseline Recall@1: {baseline_metrics.get(f'{prefix}recall@1', 0):.4f}")
    print(f"  Baseline Recall@5: {baseline_metrics.get(f'{prefix}recall@5', 0):.4f}")

    # 4. Prepare Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(train_pairs) + batch_size - 1) // batch_size
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.1 * total_training_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    print(f"\nTraining plan: {steps_per_epoch} steps/epoch x {epochs} epochs = {total_training_steps} total steps (Warmup: {warmup_steps} steps).")

    # 5. Training Loop
    use_amp = (device == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    history = []
    best_ndcg = baseline_metrics.get(f"{prefix}ndcg@10", 0.0)
    best_epoch = 0
    best_state_dict = None

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # Shuffle training pairs reproducibly per epoch
        epoch_indices = list(range(len(train_pairs)))
        random.Random(seed + epoch).shuffle(epoch_indices)
        
        epoch_loss = 0.0
        step_losses = []

        for step in range(steps_per_epoch):
            batch_idxs = epoch_indices[step * batch_size : (step + 1) * batch_size]
            batch_pairs = [train_pairs[idx] for idx in batch_idxs]

            batch_queries = [p["query"] for p in batch_pairs]
            batch_passages = [p["passage"] for p in batch_pairs]

            # Tokenize / preprocess features
            features_q = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in model.preprocess(batch_queries).items()
            }
            features_p = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in model.preprocess(batch_passages).items()
            }

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                loss = loss_fct([features_q, features_p], labels=None)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step_losses.append(loss_val)

        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_duration = time.perf_counter() - epoch_start

        # Evaluation at epoch end
        model.eval()
        epoch_eval = evaluator(model)
        val_ndcg = epoch_eval.get(f"{prefix}ndcg@10", 0.0)
        val_mrr = epoch_eval.get(f"{prefix}mrr@10", 0.0)
        val_rec1 = epoch_eval.get(f"{prefix}recall@1", 0.0)
        val_rec5 = epoch_eval.get(f"{prefix}recall@5", 0.0)

        epoch_record = {
            "epoch": epoch,
            "avg_loss": round(avg_epoch_loss, 5),
            "step_losses": [round(l, 5) for l in step_losses],
            "val_ndcg@10": round(val_ndcg, 4),
            "val_mrr@10": round(val_mrr, 4),
            "val_recall@1": round(val_rec1, 4),
            "val_recall@5": round(val_rec5, 4),
            "epoch_duration_seconds": round(epoch_duration, 2),
            "current_lr": round(scheduler.get_last_lr()[0], 7),
        }
        history.append(epoch_record)

        is_best = val_ndcg > best_ndcg
        if is_best:
            best_ndcg = val_ndcg
            best_epoch = epoch
            # Cache state dict of best model
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        best_marker = " [* NEW BEST]" if is_best else ""
        print(
            f"Epoch {epoch}/{epochs} | Avg Loss: {avg_epoch_loss:.4f} | "
            f"Val nDCG@10: {val_ndcg:.4f} | MRR@10: {val_mrr:.4f} | "
            f"Recall@1: {val_rec1:.4f} | Recall@5: {val_rec5:.4f} | "
            f"Time: {epoch_duration:.1f}s{best_marker}"
        )

    total_duration = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_duration:.2f} seconds.")

    # 6. Load Best Model Checkpoint
    if best_state_dict is not None:
        print(f"Loading best checkpoint from Epoch {best_epoch} (nDCG@10: {best_ndcg:.4f})...")
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    else:
        print("Using final model weights (no epoch surpassed pre-training baseline).")

    # 7. Final Validation Evaluation on Best Model
    model.eval()
    final_val_eval = evaluator(model)

    final_ndcg = final_val_eval.get(f"{prefix}ndcg@10", 0.0)
    final_mrr = final_val_eval.get(f"{prefix}mrr@10", 0.0)
    final_rec1 = final_val_eval.get(f"{prefix}recall@1", 0.0)
    final_rec5 = final_val_eval.get(f"{prefix}recall@5", 0.0)
    final_rec10 = final_val_eval.get(f"{prefix}recall@10", 0.0)
    final_acc1 = final_val_eval.get(f"{prefix}accuracy@1", 0.0)
    final_acc5 = final_val_eval.get(f"{prefix}accuracy@5", 0.0)
    final_acc10 = final_val_eval.get(f"{prefix}accuracy@10", 0.0)
    final_map100 = final_val_eval.get(f"{prefix}map@100", 0.0)

    delta_ndcg = final_ndcg - baseline_metrics.get(f"{prefix}ndcg@10", 0.0)
    delta_mrr = final_mrr - baseline_metrics.get(f"{prefix}mrr@10", 0.0)
    delta_rec1 = final_rec1 - baseline_metrics.get(f"{prefix}recall@1", 0.0)
    delta_rec5 = final_rec5 - baseline_metrics.get(f"{prefix}recall@5", 0.0)

    verdict = "WIN" if delta_ndcg >= 0.005 else "MARGINAL" if delta_ndcg >= 0 else "REGRESSION"

    # 8. Save Fine-Tuned Model
    if output_model_dir is None:
        output_model_dir = MODELS_DIR / "fine_tuned_all_minilm_l6_v2"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving fine-tuned model to new directory: {output_model_dir}")
    model.save(str(output_model_dir))
    print(f"Model saved successfully to {output_model_dir}")

    # 9. Save Validation Benchmark Results
    if results_file is None:
        results_file = RESULTS_DIR / "finetuned_val_metrics.json"

    # Ensure never overwriting baseline results
    if results_file.name == "baseline_val_metrics.json":
        raise ValueError("Cannot overwrite baseline_val_metrics.json! Use finetuned_val_metrics.json.")

    if results_file.exists():
        timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_file.with_name(f"{results_file.stem}_{timestamp_suffix}{results_file.suffix}")

    report = {
        "evaluation_name": "post_finetuning_validation_evaluation",
        "verdict": verdict,
        "base_model": model_name_or_path,
        "fine_tuned_model_dir": str(output_model_dir),
        "split": "val",
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device,
        "gpu_name": gpu_name,
        "training_configuration": {
            "loss_function": "MultipleNegativesRankingLoss",
            "mnrl_scale": scale,
            "mnrl_temperature": 1.0 / scale,
            "train_pairs_count": len(train_pairs),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "total_steps": total_training_steps,
            "seed": seed,
            "precision": "bfloat16" if use_amp else "float32",
            "tf32": True if device == "cuda" else False,
            "total_duration_seconds": round(total_duration, 2),
            "best_epoch": best_epoch,
        },
        "training_loss_trajectory": {
            "initial_step_loss": history[0]["step_losses"][0],
            "final_step_loss": history[-1]["step_losses"][-1],
            "epochs": history,
        },
        "validation_metrics": {
            "baseline": {
                "nDCG@10": baseline_metrics.get(f"{prefix}ndcg@10", 0.0),
                "MRR@10": baseline_metrics.get(f"{prefix}mrr@10", 0.0),
                "Recall@1": baseline_metrics.get(f"{prefix}recall@1", 0.0),
                "Recall@5": baseline_metrics.get(f"{prefix}recall@5", 0.0),
                "Recall@10": baseline_metrics.get(f"{prefix}recall@10", 0.0),
                "Accuracy@1": baseline_metrics.get(f"{prefix}accuracy@1", 0.0),
                "Accuracy@5": baseline_metrics.get(f"{prefix}accuracy@5", 0.0),
                "Accuracy@10": baseline_metrics.get(f"{prefix}accuracy@10", 0.0),
                "MAP@100": baseline_metrics.get(f"{prefix}map@100", 0.0),
            },
            "finetuned": {
                "nDCG@10": final_ndcg,
                "MRR@10": final_mrr,
                "Recall@1": final_rec1,
                "Recall@5": final_rec5,
                "Recall@10": final_rec10,
                "Accuracy@1": final_acc1,
                "Accuracy@5": final_acc5,
                "Accuracy@10": final_acc10,
                "MAP@100": final_map100,
            },
            "deltas": {
                "nDCG@10": delta_ndcg,
                "MRR@10": delta_mrr,
                "Recall@1": delta_rec1,
                "Recall@5": delta_rec5,
            },
        },
        "raw_evaluator_metrics": {k: float(v) for k, v in final_val_eval.items()},
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("PHASE 3 FINE-TUNING & VALIDATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"VERDICT:              {verdict}")
    print(f"Training Duration:    {total_duration:.2f} s")
    print(f"Initial Step Loss:    {history[0]['step_losses'][0]:.4f}")
    print(f"Final Step Loss:      {history[-1]['step_losses'][-1]:.4f}")
    print("-" * 75)
    print(f"{'Metric':<15} | {'Baseline':<12} | {'Fine-Tuned':<12} | {'Delta':<12}")
    print("-" * 75)
    print(f"{'nDCG@10':<15} | {baseline_metrics.get(f'{prefix}ndcg@10', 0):<12.4f} | {final_ndcg:<12.4f} | {delta_ndcg:+12.4f}")
    print(f"{'MRR@10':<15} | {baseline_metrics.get(f'{prefix}mrr@10', 0):<12.4f} | {final_mrr:<12.4f} | {delta_mrr:+12.4f}")
    print(f"{'Recall@1':<15} | {baseline_metrics.get(f'{prefix}recall@1', 0):<12.4f} | {final_rec1:<12.4f} | {delta_rec1:+12.4f}")
    print(f"{'Recall@5':<15} | {baseline_metrics.get(f'{prefix}recall@5', 0):<12.4f} | {final_rec5:<12.4f} | {delta_rec5:+12.4f}")
    print("-" * 75)
    print(f"Fine-tuned model saved at:  {output_model_dir}")
    print(f"Validation metrics saved at: {results_file}")
    print("=" * 75)

    return report


def main():
    parser = argparse.ArgumentParser(description="Contrastive Fine-Tuning with MNRL.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model-dir", type=str, default=None)
    parser.add_argument("--results-file", type=str, default=None)

    args = parser.parse_args()

    out_dir = Path(args.output_model_dir) if args.output_model_dir else None
    res_file = Path(args.results_file) if args.results_file else None

    train_model(
        model_name_or_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scale=args.scale,
        seed=args.seed,
        output_model_dir=out_dir,
        results_file=res_file,
    )


if __name__ == "__main__":
    main()
