import random
import math
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Value(self.data * -1, (self,), '-')

        def _backward():
            self.grad += (-1 * out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), '**')

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self**-1)
    
    
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self,x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

neuron = Neuron(2)

x = [Value(2.0), Value(3.0)]
output = neuron(x)

layer = Layer(2, 3)

x = [Value(2), Value(3)]
outputs = layer(x)

print("Outputs:", outputs)
print("Number of outputs:", len(outputs))

loss = sum(outputs, Value(0))
loss.backward()


print("First neuron's weight gradients:",
      [w.grad for w in layer.neurons[0].w])


print("Output:", output.data)
print("Weights:", neuron.w)
print("Bias:", neuron.b)


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i+1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]


# Create the training dataset
training_data = [
    ([2.0, 3.0, -1.0], 1.0),
    ([3.0, -1.0, 0.5], -1.0),
    ([-1.0, -2.0, 2.0], 1.0),
    ([1.0, 1.0, -1.0], -1.0),
]

# Create the network
network = MLP(3, [4, 4, 1])

# Training settings
learning_rate = 0.05
epochs = 100

for epoch in range(epochs):

    # Calculate predictions and loss
    losses = []

    for inputs, target_value in training_data:
        x = [Value(value) for value in inputs]
        prediction = network(x)
        target = Value(target_value)

        loss = (prediction - target) ** 2
        losses.append(loss)

    total_loss = sum(losses, Value(0))

    # Reset gradients before backpropagation
    for parameter in network.parameters():
        parameter.grad = 0

    # Calculate gradients
    total_loss.backward()

    # Update weights and biases
    for parameter in network.parameters():
        parameter.data -= learning_rate * parameter.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data:.4f}")


import torch

# Custom autograd engine
x_custom = Value(2.0)
y_custom = Value(-3.0)

z_custom = x_custom**2 + x_custom * y_custom + y_custom**2
z_custom.backward()

# PyTorch autograd
x_torch = torch.tensor(2.0, requires_grad=True)
y_torch = torch.tensor(-3.0, requires_grad=True)

z_torch = x_torch**2 + x_torch * y_torch + y_torch**2
z_torch.backward()

# Compare results
print("\n--- Gradient Comparison ---")
print(f"Custom output: {z_custom.data}")
print(f"PyTorch output: {z_torch.item()}")

print(f"Custom dx: {x_custom.grad}")
print(f"PyTorch dx: {x_torch.grad.item()}")

print(f"Custom dy: {y_custom.grad}")
print(f"PyTorch dy: {y_torch.grad.item()}")