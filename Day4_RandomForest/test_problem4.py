"""
Tests for Day 4 — Random Forest (from scratch)
Run:
    python test_problem.py
"""
import numpy as np
from problem4 import RandomForestScratch

def test_separable_blobs():
    rng = np.random.default_rng(1)
    X0 = rng.normal([-2, -2], 0.8, (250, 2))
    X1 = rng.normal([ 2,  2], 0.8, (250, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(250, int), np.ones(250, int)])
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    split = int(0.8 * len(y)); Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]

    rf = RandomForestScratch(n_trees=30, max_depth=6, max_features="sqrt", random_state=7).fit(Xtr, ytr)
    acc = (rf.predict(Xte) == yte).mean()
    print("Accuracy (separable):", acc)
    assert acc >= 0.93  # strong margin on separable blobs

def test_imbalanced():
    rng = np.random.default_rng(2)
    n0, n1 = 900, 100
    X0 = rng.normal([0, 0], 1.0, (n0, 2))
    X1 = rng.normal([2, 2], 1.0, (n1, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, int), np.ones(n1, int)])
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    split = int(0.8 * len(y)); Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]

    rf = RandomForestScratch(n_trees=40, max_depth=8, max_features="sqrt", random_state=9).fit(Xtr, ytr)
    yhat = rf.predict(Xte)
    acc = (yhat == yte).mean()
    baseline = max(np.mean(yte == 0), np.mean(yte == 1))
    print(f"Baseline={baseline:.3f}  RF acc={acc:.3f}")
    assert acc >= baseline - 1e-6  # at least match or exceed majority baseline

if __name__ == "__main__":
    test_separable_blobs()
    test_imbalanced()
    print("All tests passed.")
