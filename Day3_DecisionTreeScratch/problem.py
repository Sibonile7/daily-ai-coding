"""
Decision Tree Classifier from Scratch (NumPy Only)
-------------------------------------------------
Goal: Build a binary decision tree classifier using Gini impurity
Author: Sibonile — Daily AI Coding Challenge (Week 1, Day 3)
"""

import numpy as np


def _gini(y):
    """Compute Gini impurity: 1 - p1² - p0²"""
    if y.size == 0:
        return 0.0
    p1 = np.mean(y == 1) #calculates the function of items that are in class 1
    return 1.0 - p1**2 - (1 - p1)**2


def _leaf_value(y):
    """Return majority class."""
    p1 = np.mean(y == 1)
    return 1 if p1 >= 0.5 else 0


class _Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "pred")

    def __init__(self, is_leaf=False, feature=None, threshold=None, left=None, right=None, pred=None):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.pred = pred


class DecisionTreeScratch:
    """Simple Decision Tree Classifier (binary)"""

    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        n, d = X.shape
        best_gain, best_feature, best_threshold, best_mask = 0.0, None, None, None
        parent_imp = _gini(y)

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for thr in thresholds:
                mask = X[:, j] <= thr
                yL, yR = y[mask], y[~mask]
                if yL.size == 0 or yR.size == 0:
                    continue
                gL, gR = _gini(yL), _gini(yR)
                gain = parent_imp - (yL.size/n)*gL - (yR.size/n)*gR
                if gain > best_gain:
                    best_gain, best_feature, best_threshold, best_mask = gain, j, thr, mask
        return best_gain, best_feature, best_threshold, best_mask

    def _build_tree(self, X, y, depth):
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or _gini(y) == 0.0
        ):
            return _Node(is_leaf=True, pred=_leaf_value(y))

        gain, feat, thr, mask = self._best_split(X, y)
        if gain < self.min_impurity_decrease or feat is None:
            return _Node(is_leaf=True, pred=_leaf_value(y))

        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return _Node(is_leaf=False, feature=feat, threshold=thr, left=left, right=right)

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.pred

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])
