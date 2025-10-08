"""
logistic_regression_scratch.py
--------------------------------
NumPy-only implementation of binary Logistic Regression with:
- Gradient Descent optimizer
- L2 regularization
- Class weighting for imbalance
- Stable sigmoid
- Early stopping by tolerance on loss
- Threshold-based prediction helper

Author: Sibonile — Daily AI Coding Challenge Series
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple

Array = np.ndarray


def _sigmoid(z: Array) -> Array:
    """Numerically stable sigmoid by clamping logits."""
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _add_bias(X: Array) -> Array:
    """Add bias column of ones."""
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def _weighted_log_loss(
    y_true: Array,
    y_prob: Array,
    sample_weight: Optional[Array] = None,
    lambda_: float = 0.0,
    w: Optional[Array] = None,
) -> float:
    """
    Weighted binary cross-entropy + L2 penalty (bias excluded).
    y_true: shape (n,) with values in {0,1}
    y_prob: shape (n,) probabilities in (0,1)
    sample_weight: shape (n,) or None
    """
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    # Mean weighted log loss
    loss = - (sample_weight * (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))).mean()
    if w is not None and lambda_ > 0:
        # exclude bias term from regularization
        loss += (lambda_ / 2.0) * np.dot(w[1:], w[1:]) / y_true.shape[0]
    return float(loss)


class LogisticRegressionScratch:
    """
    Binary Logistic Regression (NumPy-only).

    Parameters
    ----------
    lr : float
        Learning rate for gradient descent.
    epochs : int
        Maximum number of epochs.
    lambda_ : float
        L2 regularization strength.
    tol : float or None
        Convergence tolerance on absolute loss improvement. If None, disables early stopping.
    class_weight : dict or None
        e.g., {'0': w0, '1': w1} to rebalance classes.
    random_state : int
        Seed for weight initialization.
    verbose : bool
        If True, prints periodic loss updates.
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 2000,
        lambda_: float = 0.0,
        tol: Optional[float] = 1e-6,
        class_weight: Optional[Dict[str, float]] = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.w: Optional[Array] = None  # includes bias at index 0

    # ---------------------------- API ----------------------------
    def fit(self, X: Array, y: Array) -> "LogisticRegressionScratch":
        """
        Fit model using batch gradient descent.
        X: (n, d) array-like
        y: (n,) labels in {0,1}
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert set(np.unique(y)).issubset({0.0, 1.0}), "y must be binary {0,1}"

        Xb = _add_bias(X)
        n, d_plus_bias = Xb.shape
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(scale=0.01, size=d_plus_bias)

        # sample weights from class_weight
        if self.class_weight is not None:
            w0 = float(self.class_weight.get('0', 1.0))
            w1 = float(self.class_weight.get('1', 1.0))
            sw = np.where(y == 1.0, w1, w0).astype(float)
        else:
            sw = np.ones_like(y, dtype=float)

        prev_loss = np.inf
        for epoch in range(self.epochs):
            z = Xb @ self.w
            p = _sigmoid(z)

            # gradient of weighted log loss + L2 (exclude bias)
            self.w = rng.normal(scale=0.01, size=d_plus_bias)   # shape (d+1,)
            assert self.w is not None                            # silence Optional warning
            w = self.w  

            residual = (p - y) * sw                 # (n,)
            grad = (Xb.T @ residual) / n            # (d+1,)
            grad[1:] += self.lambda_ * w[1:] / n    # L2 on weights (exclude bias)
            self.w -= self.lr * grad
      
            # loss & early stop
            if self.tol is not None or self.verbose:
                loss = _weighted_log_loss(y, p, sample_weight=sw, lambda_=self.lambda_, w=self.w)
                if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                    print(f"epoch={epoch:4d} loss={loss:.6f}")
                if self.tol is not None and abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        return self

    def predict_proba(self, X: Array) -> Array:
        """Return probabilities P(y=1|x)."""
        assert self.w is not None, "Model is not fit yet."
        X = np.asarray(X, dtype=float)
        Xb = _add_bias(X)
        return _sigmoid(Xb @ self.w)

    def predict(self, X: Array) -> Array:
        """Return labels {0,1} using threshold 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # Stretch helper
    def predict_threshold(self, X: Array, t: float = 0.5) -> Array:
        """Return labels using custom probability threshold t."""
        return (self.predict_proba(X) >= float(t)).astype(int)


# ---------------------------- Utilities ----------------------------
def accuracy(y_true: Array, y_pred: Array) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float((y_true == y_pred).mean())


def precision_recall_thresholds(
    y_true: Array, y_score: Array, thresholds: Optional[Array] = None
) -> Tuple[Array, Array, Array]:
    """
    Compute precision & recall over thresholds in [0,1].
    Returns: thresholds, precisions, recalls
    """
    y_true = y_true.astype(int).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls = [], []
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fp = np.sum((y_true == 0) & (y_hat == 1))
        fn = np.sum((y_true == 1) & (y_hat == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec); recalls.append(rec)
    return thresholds, np.array(precisions), np.array(recalls)


# ---------------------------- Quick demo ----------------------------
if __name__ == "__main__":
    # Synthetic dataset (imbalanced)
    rng = np.random.default_rng(0)
    n0, n1 = 600, 200
    X0 = rng.normal([0.0, 0.0], 1.0, size=(n0, 2))
    X1 = rng.normal([2.0, 2.0], 1.0, size=(n1, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Train / test split
    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    # Majority baseline on test
    baseline = max(np.mean(yte == 0), np.mean(yte == 1))

    # Inverse-frequency class weights
    w0 = 0.5 / (np.mean(ytr == 0) + 1e-12)
    w1 = 0.5 / (np.mean(ytr == 1) + 1e-12)

    model = LogisticRegressionScratch(
        lr=0.1, epochs=5000, lambda_=0.01, tol=1e-7,
        class_weight={'0': w0, '1': w1}, verbose=False
    ).fit(Xtr, ytr)

    probs = model.predict_proba(Xte)
    yhat = (probs >= 0.5).astype(int)
    acc = accuracy(yte, yhat)

    print(f"Baseline acc (majority): {baseline:.3f}")
    print(f"Model acc: {acc:.3f}")

    # Optional: best F1 threshold
    thresholds, precs, recs = precision_recall_thresholds(yte, probs)
    f1s = 2 * (precs * recs) / (precs + recs + 1e-12)
    best_idx = int(np.argmax(f1s))
    print(f"Best F1 threshold: {thresholds[best_idx]:.2f}  "
          f"precision={precs[best_idx]:.3f}  recall={recs[best_idx]:.3f}")
