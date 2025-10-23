"""
svm_hard_margin.py
Hard-margin linear SVM (NumPy-only) with a helper to plot decision boundary and margins.
Use on linearly separable data. For non-separable data, switch to soft margin (reduce C).
"""

import numpy as np
import matplotlib.pyplot as plt

def _add_bias(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

class LinearSVMHard:
    """
    Solve: minimize 0.5 * ||w||^2  subject to  y_i * (w·x_i + b) >= 1  for all i.
    We approximate constraints with a very large hinge penalty (C) and stop once all margins >= 1.
    Bias is w[0]; we do not regularize the bias term.
    """
    def __init__(self, lr=0.05, epochs=20000, C=1e5, tol=1e-6, verbose=False, random_state=0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.w = None
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert set(np.unique(y)).issubset({-1.0, 1.0}), "Labels must be in {-1, +1}"

        # Standardize features for stable convergence
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-12
        Xs = (X - self.mu_) / self.sigma_

        Xb = _add_bias(Xs)
        n, d = Xb.shape
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(scale=0.01, size=d)

        prev_obj = np.inf
        for epoch in range(self.epochs):
            margins = y * (Xb @ self.w)
            viol = margins < 1.0

            # Objective (for monitoring)
            obj = 0.5 * np.dot(self.w[1:], self.w[1:])
            if np.any(viol):
                obj += self.C * np.mean(1.0 - margins[viol])

            # Gradient
            grad = np.zeros_like(self.w)
            grad[1:] += self.w[1:]  # L2 regularization (not bias)
            if np.any(viol):
                Xv, yv = Xb[viol], y[viol]
                grad += -(self.C / max(1, yv.size)) * (Xv.T @ yv)

            # Weight update
            self.w -= self.lr * grad

            # Stop if all constraints satisfied
            if np.all(margins >= 1.0 - 1e-8):
                if self.verbose:
                    print(f"[SVM] Converged at epoch {epoch}")
                break

            prev_obj = obj
        else:
            raise RuntimeError("Data appears non-separable: hard-margin constraints not satisfied.")
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs = (X - self.mu_) / self.sigma_
        return _add_bias(Xs) @ self.w

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)


def plot_hard_margin_2d(model, X, y, title="Hard-Margin SVM"):
    """
    Plot decision boundary (solid) and margins (dashed) for 2D features.
    """
    X = np.asarray(X, dtype=float)
    assert X.shape[1] == 2, "Plotting only supports exactly 2 features."

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(7,6))
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--','-','--'])
    plt.scatter(X[:,0], X[:,1], c=(y>0).astype(int), edgecolor='k', alpha=0.9)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo on synthetic separable data
    rng = np.random.default_rng(0)
    n = 300
    X_pos = rng.normal([2.5, 2.5], [0.6, 0.6], size=(n//2, 2))
    X_neg = rng.normal([-2.5, -2.5], [0.6, 0.6], size=(n//2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n//2), -np.ones(n//2)])
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    svm = LinearSVMHard(lr=0.05, epochs=20000, C=1e5).fit(X, y)
    print(f"Accuracy: {(svm.predict(X) == y).mean():.3f}")
    plot_hard_margin_2d(svm, X, y)
