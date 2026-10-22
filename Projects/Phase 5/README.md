# Contrastive Embedding Fine-Tuning

## What this does

Fine-tunes the `all-MiniLM-L6-v2` sentence embedding model using contrastive learning with `MultipleNegativesRankingLoss`. Evaluates retrieval performance before and after fine-tuning and visualizes the embedding changes using PCA.

## Problems/challenge this solves

* Adapts a pretrained embedding model to a small collection of AI and machine-learning concepts.
* Trains the model to place related queries and text closer in embedding space.
* Measures retrieval performance using Recall@1, Recall@3, and Mean Reciprocal Rank (MRR).
* Compares the original and fine-tuned embeddings visually using PCA.

## Key decisions

* Used `all-MiniLM-L6-v2` as the pretrained baseline to avoid training an embedding model from scratch.
* Used anchor-positive training pairs with `MultipleNegativesRankingLoss`, which treats other examples in the batch as negatives.
* Evaluated retrieval against a separate set of six queries and twelve documents.
* Used normalized embeddings and dot-product similarity for retrieval ranking.
* Used Recall@1, Recall@3, and MRR to evaluate retrieval quality at different ranking positions.
* Fitted PCA on the original embeddings and reused the same transformation for the fine-tuned embeddings to make the visualization comparable.

## How to run

```bash
python contrastive_embedding_finetuning.py
```

## Results

The script reports retrieval metrics before and after fine-tuning:

* Recall@1
* Recall@3
* Mean Reciprocal Rank (MRR)

It also generates a PCA scatter plot comparing embeddings before and after fine-tuning.

Actual before-and-after metric values should be added after running the script.
