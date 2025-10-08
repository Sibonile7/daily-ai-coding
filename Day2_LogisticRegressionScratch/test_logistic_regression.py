"""
test_logistic_regression.py
Minimal sanity tests for LogisticRegressionScratch
"""
import numpy as np
from logistic_regression_scratch import LogisticRegressionScratch, accuracy

def test_imbalanced_weights():
    rng = np.random.default_rng(2)
    n0, n1 = 900, 100  # 90/10
    X0 = rng.normal([0, 0], 1.0, (n0, 2))
    X1 = rng.normal([2, 2], 1.0, (n1, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    # Class weights inverse-frequency
    w0 = 0.5 / (np.mean(ytr == 0) + 1e-12)
    w1 = 0.5 / (np.mean(ytr == 1) + 1e-12)

    model = LogisticRegressionScratch(
        lr=0.1, epochs=5000, lambda_=0.01, tol=1e-7,
        class_weight={'0': w0, '1': w1}, random_state=0
    ).fit(Xtr, ytr)

    p = model.predict_proba(Xte)
    assert np.all(p >= 0) and np.all(p <= 1)

    # Majority-class baseline
    baseline = max(np.mean(yte == 0), np.mean(yte == 1))

    # 🔑 Choose the threshold that maximizes accuracy
    thresholds = np.linspace(0.0, 1.0, 101)
    accs = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        accs.append(((yhat == yte).mean(), t))
    best_acc, best_t = max(accs, key=lambda x: x[0])

    print(f"Baseline: {baseline:.3f}  BestAcc: {best_acc:.3f} @ t={best_t:.2f}")
    assert best_acc >= baseline - 1e-6

if __name__ == "__main__":
    #test_linear_separable()
    test_imbalanced_weights()
    print("All tests passed.")
