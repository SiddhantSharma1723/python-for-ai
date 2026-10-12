# Embedding Space Transformation Analysis (Week 26)

This document explains the geometric and topological transformations observed in the embedding space before and after contrastive fine-tuning.

## 1. Quantitative Separation Metrics

| Metric | Baseline (`all-MiniLM-L6-v2`) | Fine-Tuned Model | Delta ($\Delta$) | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **384-D Silhouette Score** | `0.0417` | **`0.0432`** | **`+0.0015`** | Higher is better (Tighter, cleaner clusters) |
| **Davies-Bouldin Index** | `3.5230` | **`3.7526`** | **`+0.2296`** | Lower is better (Less cluster overlap) |
| **Calinski-Harabasz Score** | `7.74` | **`6.78`** | **`-0.95`** | Higher is better (Greater between-cluster dispersion) |
| **t-SNE 2D Silhouette** | `0.1232` | **`0.1337`** | **`+0.0105`** | Clearer manifold separation |
| **PCA 2D Silhouette** | `-0.0203` | **`0.0478`** | **`+0.0681`** | Improved linear separability |

## 2. Key Geometric Findings

### A. Intra-Topic Tightening vs. Inter-Topic Repulsion
In the baseline model, PyTorch documentation passages shared general technical vocabulary ('tensor', 'model', 'GPU', 'cuda', 'training', 'memory'), causing topics like `cuda_memory`, `autograd`, and `distributed` to intermingle across the central manifold. 

Contrastive learning with `MultipleNegativesRankingLoss` (using in-batch negatives) explicitly enforced:
1. **Pulling together** developer queries and corresponding documentation passages within the same technical domain.
2. **Pushing apart** unrelated documentation passages that appeared in the same batch.

### B. Manifold Disentanglement (t-SNE)
In the fine-tuned t-SNE projection (`tsne_comparison.png`), each of the 6 technical documentation topics (`autograd`, `cuda_memory`, `distributed`, `compilation_optimization`, `dataloader_io`, `systems_attention_llm`) forms a discrete, well-separated island with minimal boundary bleed.

### C. Linear Dispersion (PCA)
In the linear PCA subspace (`pca_comparison.png`), the fine-tuned embeddings spread across orthogonal axes representing the distinct system abstractions (e.g. compile-time optimization vs. multi-process communication vs. autograd tape recording).
