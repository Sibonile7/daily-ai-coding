"""
experiment_logistic_regression.py
---------------------------------
Run experiments on your LogisticRegressionScratch model:
- threshold tuning (precision/recall/F1 vs threshold)
- regularization sweep (lambda_)
- learning rate sweep (lr)
- decision boundary visualization
"""

import os
import numpy as np

# Use a safe backend if no display (prevents crashes on headless runs)
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from logistic_regression_scratch import (
    LogisticRegressionScratch,
    precision_recall_thresholds,
    accuracy,
)

# ---------- Helpers ----------
def standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-12
    return (X - mu) / sigma, mu, sigma

def make_synthetic(n0=600, n1=200, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([0.0, 0.0], 1.0, (n0, 2))
    X1 = rng.normal([2.0, 2.0], 1.0, (n1, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.2):
    n = len(y)
    split = int((1 - test_size) * n)
    return X[:split], X[split:], y[:split], y[split:]

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    # 1) Data
    X, y = make_synthetic()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    # Standardize (helps convergence/stability)
    Xtr, mu, sigma = standardize(Xtr)
    Xte = (Xte - mu) / sigma

    # Class weights (optional but helpful)
    w0 = 0.5 / (np.mean(ytr == 0) + 1e-12)
    w1 = 0.5 / (np.mean(ytr == 1) + 1e-12)
    class_weight = {'0': w0, '1': w1}

    # 2) Train base model
    model = LogisticRegressionScratch(
        lr=0.1, epochs=5000, lambda_=0.01, tol=1e-7, class_weight=class_weight
    ).fit(Xtr, ytr)

    # 3) Threshold sweep: Precision, Recall, F1
    probs = model.predict_proba(Xte)
    thresholds, precisions, recalls = precision_recall_thresholds(yte, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)

    plt.figure()
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, '--', label='F1 Score')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title('Precision/Recall/F1 vs Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    best_idx = int(np.argmax(f1s))
    print(f"Best threshold = {thresholds[best_idx]:.2f}, F1 = {f1s[best_idx]:.3f}")

    # 4) Regularization sweep (with class weights)
    print("[Regularization sweep]")
    for lam in [0.0, 0.01, 0.1, 1.0]:
        m = LogisticRegressionScratch(
            lambda_=lam, lr=0.1, epochs=4000, tol=1e-7, class_weight=class_weight, random_state=0
        ).fit(Xtr, ytr)
        acc_val = accuracy(yte, m.predict(Xte))
        print(f"lambda={lam:<5} -> Accuracy={acc_val:.3f}")

    # 5) Learning-rate sweep
    print("[Learning-rate sweep]")
    for lr in [0.001, 0.01, 0.1, 0.5]:
        m = LogisticRegressionScratch(
            lr=lr, epochs=2000, lambda_=0.01, tol=1e-7, class_weight=class_weight, random_state=1
        ).fit(Xtr, ytr)
        acc_val = accuracy(yte, m.predict(Xte))
        print(f"lr={lr:<6} -> Accuracy={acc_val:.3f}")

    # 6) Decision boundary
    plot_decision_boundary(model, Xte, yte, title="Decision Boundary (Test)")
