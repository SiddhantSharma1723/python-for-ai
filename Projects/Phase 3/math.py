import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.linspace(0,10,100)
y = (x>5).astype(int)

w = 0
b = 0
lr = 0.001
epochs = 5000

losses = []
for i in range(epochs):
    z = w*x + b
    y_pred = 1 / (1 + np.exp(-z))

    loss = -np.mean(y*np.log(y_pred) + (1-y)*(np.log(1-y_pred)))
    losses.append(loss)

    dw = np.mean(x*(y_pred - y))
    db = np.mean(y_pred - y)

    w -= lr*dw
    b -= lr*db

    if i % 500 == 0:
        print(f"epochs = {i}, loss = {loss:.3f}, w = {w:.3f}, b = {b:.3f}")
print(w, b)

plt.plot(losses)
plt.title("losses")
plt.show()

plt.scatter(x,y)
plt.plot(x, 1 / (1+ np.exp(-(w*x + b))), color = "crimson")
plt.title("regression")
plt.show()