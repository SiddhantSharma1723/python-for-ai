import torch
import torch.nn as nn

# Make results reproducible
torch.manual_seed(42)

# Training data
X = torch.tensor(
    [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [-1.0, -2.0, 2.0],
        [1.0, 1.0, -1.0],
    ],
    dtype=torch.float32,
)

y = torch.tensor(
    [[1.0], [-1.0], [1.0], [-1.0]],
    dtype=torch.float32,
)

# Network architecture: 3 -> 4 -> 4 -> 1
network = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Tanh(),
)

# Training setup
optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
loss_function = nn.MSELoss()

# Train
for epoch in range(100):
    predictions = network(X)
    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("\nFinal predictions:")
print(network(X).detach())