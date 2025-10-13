import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from problem1 import DecisionTreeScratch

def test_simple_split():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeScratch(max_depth=2).fit(X, y)
    yhat = tree.predict(X)
    acc = (yhat == y).mean()
    print("Accuracy:", acc)
    assert acc >= 0.9

def test_random_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 1, (50, 2))
    X1 = rng.normal([2, 2], 1, (50, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(50), np.ones(50)])
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    tree = DecisionTreeScratch(max_depth=4).fit(Xtr, ytr)
    acc = (tree.predict(Xte) == yte).mean()
    print("Test accuracy:", acc)
    assert acc > 0.8

if __name__ == "__main__":
    test_simple_split()
    test_random_data()
    print("✅ All tests passed.")
