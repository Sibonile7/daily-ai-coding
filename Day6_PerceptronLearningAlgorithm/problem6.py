"""
Day 8 — Perceptron Learning Algorithm (PLA)
-------------------------------------------
Implementation of a Perceptron classifier using NumPy:
- Train on 2D separable data.
- Update weights using PLA rule.
- Visualize decision boundary evolution.
- Report number of updates until convergence.
"""

import numpy as np
import matplotlib.pyplot as plt


def _add_bias(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


class PerceptronPLA:
    """Binary Perceptron for linearly separable data."""

    def __init__(self, lr=1.0, max_epochs=1000, random_state=42):
        self.lr = lr
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.w_ = None
        self.updates_ = 0
        self.converged_ = False

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Xb = _add_bias(X)
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(scale=0.01, size=Xb.shape[1])

        for _ in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(Xb, y):
                if yi * np.dot(self.w_, xi) <= 0:
                    self.w_ += self.lr * yi * xi
                    self.updates_ += 1
                    errors += 1
            if errors == 0:
                self.converged_ = True
                break
        return self

    def predict(self, X):
        Xb = _add_bias(np.asarray(X, float))
        return np.where(Xb @ self.w_ >= 0, 1, -1)


def plot_boundary(X, y, w):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_vals = -(w[0] + w[1] * x_vals) / w[2]
    plt.plot(x_vals, y_vals, "k--")
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.show()


if __name__ == "__main__":
    # Generate synthetic separable data
    rng = np.random.default_rng(0)
    X1 = rng.normal([2, 2], 0.5, (50, 2))
    X2 = rng.normal([-2, -2], 0.5, (50, 2))
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(50), -np.ones(50)])

    model = PerceptronPLA(lr=1.0, max_epochs=1000).fit(X, y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)

    print(f"Converged: {model.converged_}")
    print(f"Total updates: {model.updates_}")
    print(f"Training accuracy: {acc:.3f}")

    plot_boundary(X, y, model.w_)
