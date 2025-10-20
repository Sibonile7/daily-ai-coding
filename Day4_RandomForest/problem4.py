"""
Day 4 — Random Forest (from scratch, NumPy-only)

Implements:
- DecisionTreeScratch (Gini, continuous splits)
- RandomForestScratch (bagging + feature subsampling)

API:
    rf = RandomForestScratch(
        n_trees=25, max_depth=6, min_samples_split=2,
        max_features="sqrt", bootstrap=True, random_state=42
    ).fit(X, y)

    y_hat = rf.predict(X_test)
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Union, List

Array = np.ndarray


# --------------------------- Decision Tree ---------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "is_leaf", "pred")
    def __init__(self, is_leaf=False, pred=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.pred = pred
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def _gini(y: Array) -> float:
    if y.size == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1.0 - p1
    return 1.0 - p1 * p1 - p0 * p0

def _leaf_value(y: Array) -> int:
    # majority class; break ties toward 0
    p1 = np.mean(y == 1)
    return 1 if p1 > 0.5 else 0

class DecisionTreeScratch:
    """
    Minimal binary decision tree (Gini, continuous features).
    """
    def __init__(self, max_depth: int = 6, min_samples_split: int = 2, min_impurity_decrease: float = 1e-7,
                 max_features: Optional[Union[int, str]] = None, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features  # int | "sqrt" | None
        self.random_state = random_state
        self.root: Optional[_Node] = None
        self.rng = np.random.default_rng(random_state)

    def _choose_features(self, d: int) -> Array:
        if self.max_features is None:
            k = d
        elif isinstance(self.max_features, int):
            k = max(1, min(d, self.max_features))
        elif isinstance(self.max_features, str) and self.max_features.lower() == "sqrt":
            k = max(1, int(np.sqrt(d)))
        else:
            k = d
        return self.rng.choice(d, size=k, replace=False)

    def fit(self, X: Array, y: Array):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=int).reshape(-1)
        assert set(np.unique(y)).issubset({0, 1})
        self.root = self._build(X, y, depth=0)
        return self

    def _best_split(self, X: Array, y: Array):
        n, d = X.shape
        parent_imp = _gini(y)
        best_gain, best_feat, best_thr, best_mask = 0.0, None, None, None

        features = self._choose_features(d)

        for j in features:
            idx = np.argsort(X[:, j], kind="mergesort")
            xj, yj = X[idx, j], y[idx]
            diffs = xj[1:] - xj[:-1]
            change = (yj[1:] != yj[:-1]) & (diffs > 0)
            if not np.any(change):
                continue
            thrs = (xj[:-1][change] + xj[1:][change]) / 2.0

            # evaluate thresholds
            for thr in thrs:
                mask = X[:, j] <= thr
                yL, yR = y[mask], y[~mask]
                if yL.size < 1 or yR.size < 1:
                    continue
                gL, gR = _gini(yL), _gini(yR)
                child_imp = (yL.size / n) * gL + (yR.size / n) * gR
                gain = parent_imp - child_imp
                if gain > best_gain:
                    best_gain, best_feat, best_thr, best_mask = gain, j, thr, mask

        return best_gain, best_feat, best_thr, best_mask

    def _build(self, X: Array, y: Array, depth: int) -> _Node:
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or _gini(y) == 0.0
        ):
            return _Node(is_leaf=True, pred=_leaf_value(y))

        gain, feat, thr, mask = self._best_split(X, y)
        if feat is None or gain < self.min_impurity_decrease:
            return _Node(is_leaf=True, pred=_leaf_value(y))

        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(is_leaf=False, feature=feat, threshold=thr, left=left, right=right)

    def _predict_one(self, x: Array) -> int:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.pred

    def predict(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self._predict_one(x) for x in X], dtype=int)


# --------------------------- Random Forest ---------------------------
class RandomForestScratch:
    """
    Random Forest with:
    - bootstrap sampling
    - feature subsampling per split (tree's max_features)
    - majority vote
    """
    def __init__(self, n_trees: int = 25, max_depth: int = 6, min_samples_split: int = 2,
                 max_features: Union[int, str, None] = "sqrt", bootstrap: bool = True,
                 random_state: Optional[int] = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees: List[DecisionTreeScratch] = []

    def fit(self, X: Array, y: Array):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=int).reshape(-1)

        n = len(y)
        self.trees = []
        for t in range(self.n_trees):
            if self.bootstrap:
                idx = self.rng.integers(0, n, size=n)  # sample with replacement
            else:
                idx = self.rng.permutation(n)
            Xb, yb = X[idx], y[idx]

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=(None if self.random_state is None else self.random_state + t),
            ).fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # collect votes
        votes = np.stack([tree.predict(X) for tree in self.trees], axis=0)  # (n_trees, n_samples)
        # majority vote (ties go to 0)
        sums = votes.sum(axis=0)
        return (sums > (len(self.trees) / 2)).astype(int)


# --------------------------- Quick Demo ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 1.0, (300, 2))
    X1 = rng.normal([2, 2], 1.0, (300, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(300, int), np.ones(300, int)])
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]

    split = int(0.8 * len(y))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    rf = RandomForestScratch(n_trees=25, max_depth=6, max_features="sqrt", random_state=0).fit(Xtr, ytr)
    acc = (rf.predict(Xte) == yte).mean()
    print(f"[Demo] RandomForestScratch accuracy: {acc:.3f}")
