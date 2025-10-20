# 🌲 Day 4 — Random Forest (from Scratch, NumPy)

## 🎯 Goal
Implement a **Random Forest Classifier** using your own **Decision Tree** as the base learner. Learn how **bagging** and **feature randomness** reduce variance and improve generalization.

## ✅ What to Build
- `DecisionTreeScratch`: Gini impurity, continuous features, binary splits `x[j] <= threshold`.
- `RandomForestScratch`:
  - `fit(X, y, n_trees=..., max_depth=..., max_features='sqrt', bootstrap=True)`
  - `predict(X)` via **majority voting** across trees.

## 🧠 Learning Bite
Random Forests average many **high-variance** trees trained on different bootstrapped samples and random feature subsets. This reduces variance (overfitting) while keeping bias reasonable — a classic **bias–variance tradeoff** win.

## 🧪 How to Run
```bash
python problem.py          # quick demo accuracy
python test_problem.py     # tests: separable blobs + imbalanced data
