import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 6],
    [5, 5],
    [6, 5],
    [7, 4],
    [8, 3],
    [9, 2],
    [10, 2]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std

m, n = X_norm.shape
weights = np.zeros(n)
bias = 0
lr = 0.1
epochs = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for epoch in range(epochs):
    z = np.dot(X_norm, weights) + bias
    y_pred = sigmoid(z)

    loss = -(1 / m) * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

    dw = (1 / m) * np.dot(X_norm.T, (y_pred - y))
    db = (1 / m) * np.sum(y_pred - y)

    weights -= lr * dw
    bias -= lr * db


def predict(X):
    z = np.dot(X, weights) + bias
    return np.where(sigmoid(z) >= 0.5, 1, 0)


y_pred_final = predict(X_norm)
print("Final Accuracy:", np.mean(y_pred_final == y))

plt.figure(figsize=(8, 6))

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label="Fail (0)")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label="Pass (1)")

x_values = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
x_norm = (x_values - X_mean[0]) / X_std[0]

if weights[1] != 0:  # avoid divide by zero
    y_values = -(weights[0] * x_norm + bias) / weights[1]
    y_values = y_values * X_std[1] + X_mean[1]
    plt.plot(x_values, y_values, color="blue", linewidth=2, label="Decision Boundary")

plt.xlabel("Hours Studied")
plt.ylabel("Sleep Hours")
plt.legend()
plt.title("Logistic Regression Decision Boundary (from scratch)")
plt.show()