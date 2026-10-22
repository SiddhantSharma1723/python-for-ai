import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sklearn.decomposition import PCA

model_name = "sentence-transformers/all-MiniLM-L6-v2"
epochs = 1
batch_size = 4
learning_rate = 2e-5

training_data = {
    "anchor": [
        "How does gradient descent update model parameters?",
        "What is the purpose of a convolutional neural network?",
        "How do transformers process sequences?",
        "What does a tokenizer do to text?",
        "How is classification accuracy calculated?",
        "What is the purpose of dropout during training?",
        "How do embeddings represent the meaning of text?",
        "What is the role of a learning rate?",
    ],
    "positive": [
        "Gradient descent updates parameters using gradients of the loss.",
        "Convolutional neural networks learn spatial features from data.",
        "Transformers use attention mechanisms to process input sequences.",
        "A tokenizer converts text into tokens or token identifiers.",
        "Classification accuracy measures the fraction of correct predictions.",
        "Dropout randomly suppresses activations during training.",
        "Embeddings encode text as numerical vectors.",
        "The learning rate controls the size of parameter updates.",
    ],
}

train_dataset = Dataset.from_dict(training_data)

documents = [
    "An optimizer updates neural network parameters using gradients to reduce a loss function.",
    "Convolutional neural networks identify patterns such as edges and shapes in images.",
    "Self-attention allows transformer models to determine relationships between tokens.",
    "Tokenization splits text into tokens that can be mapped to numerical IDs.",
    "Classification accuracy is the proportion of predictions that match the true labels.",
    "Dropout reduces overfitting by randomly disabling some activations during training.",
    "Text embeddings represent words or sentences as numerical vectors.",
    "The learning rate determines the size of each optimization step.",
    "Backpropagation calculates gradients of the loss with respect to model parameters.",
    "A validation dataset helps assess how well a model performs on data not used for training.",
    "Batch size determines how many training examples are processed in one batch.",
    "An activation function introduces nonlinearity into a neural network."
]

evaluation_queries = [
    "How are neural network weights adjusted to reduce errors?",
    "Which transformer mechanism identifies relationships between tokens?",
    "How is text split into units before entering a language model?",
    "What controls the size of an optimizer's update?",
    "Why do we use validation data during model development?",
    "What helps a neural network learn complex nonlinear patterns?"
]

relevant_indices = [0, 2, 3, 7, 9, 11]

def retrieval_documents(model, queries, corpus, top_k=3):
    queries_embeddings = model.encode(
        queries,
        convert_to_tensor = True,
        normalize_embeddings = True,
        show_progress_bar = False
    )

    document_embeddings = model.encode(
        corpus,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = queries_embeddings @ document_embeddings.T

    top_k = min(top_k, len(corpus))

    rankings = torch.argsort(
        similarities, dim=1, descending=True
    )[:,:top_k]

    return rankings.cpu().tolist()

def evaluate_retrieval(model, queries, corpus, relevant_indices):
    rankings = retrieval_documents(model, queries, corpus, top_k=3)

    recall_at_1 = []
    recall_at_3 = []
    reciprocal_rank = []

    for ranking, relevant_index in zip(rankings, relevant_indices):
        recall_at_1.append(1 if relevant_index in ranking[:1] else 0)

        recall_at_3.append(1 if relevant_index in ranking[:3] else 0)

        if relevant_index in ranking:
            rank = ranking.index(relevant_index) + 1
            reciprocal_rank.append(1 / rank)
        else:
            reciprocal_rank.append(0)
        
    return {
            "Recall@1": (sum(recall_at_1)/ len(queries)),
            "Recall@3": (sum(recall_at_3)/ len(queries)),
            "MRR": (sum(reciprocal_rank) / len(queries))
    }

model = SentenceTransformer(model_name)

before_results = evaluate_retrieval(
    model,
    evaluation_queries,
    documents,
    relevant_indices
)

print("Before fine-tuning:")

for metric, score in before_results.items():
    print(f"{metric}: {score:.4f}")


texts_to_plot = training_data["anchor"] + training_data["positive"]

before_embeddings = model.encode(
    texts_to_plot,
    normalize_embeddings=True,
    show_progress_bar=False
)

loss = MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir = "minilm-contrastive-training",
    num_train_epochs = epochs,
    per_device_train_batch_size = batch_size,
    learning_rate = learning_rate,
    warmup_steps = 0.1,
    fp16=False,
    bf16=False,
    report_to = "none",
    save_strategy = "no"
)

trainer = SentenceTransformerTrainer(
    model = model,
    args = training_args,
    loss = loss,
    train_dataset = train_dataset
)

trainer.train()
model.save("minilm-contrastive-finetuned")

after_embeddings = model.encode(
    texts_to_plot,
    normalize_embeddings=True,
    show_progress_bar=False
)

after_results = evaluate_retrieval(
    model,
    evaluation_queries,
    documents,
    relevant_indices
)

print("\nAfter fine-tuning:")
for metric, score in after_results.items():
    print(f"{metric}: {score:.4f}")

texts_to_plot = training_data["anchor"] + training_data["positive"]

embeddings = model.encode(
    texts_to_plot,
    normalize_embeddings = True,
    show_progress_bar = False
)

pca = PCA(n_components=2)

# Fit PCA on the baseline embeddings.
before_2d = pca.fit_transform(before_embeddings)

# Project the fine-tuned embeddings using the same PCA.
after_2d = pca.transform(after_embeddings)

plt.figure(figsize=(10, 7))

plt.scatter(
    before_2d[:, 0],
    before_2d[:, 1],
    label="Before fine-tuning",
    marker="o"
)

plt.scatter(
    after_2d[:, 0],
    after_2d[:, 1],
    label="After fine-tuning",
    marker="x"
)

plt.title("Embedding Comparison Before and After Fine-Tuning")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

