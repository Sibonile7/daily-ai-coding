"""
Sanity tests for LinearSVMHard on separable synthetic data.
"""
import numpy as np
from svm_hard_margin import LinearSVMHard

def test_separable_perfect():
    rng = np.random.default_rng(1)
    X_pos = rng.normal([2.5, 2.5], [0.4, 0.4], size=(150, 2))
    X_neg = rng.normal([-2.5, -2.5], [0.4, 0.4], size=(150, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(150), -np.ones(150)])
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]

    svm = LinearSVMHard(lr=0.05, epochs=15000, C=1e5).fit(X, y)
    acc = (svm.predict(X) == y).mean()
    print("Accuracy (separable):", acc)
    assert acc >= 0.99

if __name__ == "__main__":
    test_separable_perfect()
    print("All tests passed.")
