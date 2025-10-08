# 🧠 Day 2 — Binary Logistic Regression From Scratch

## 📘 Overview
In this challenge, i will implement **Logistic Regression** from scratch using **NumPy only** — no scikit-learn or high-level libraries.  
This simulates a classic **AI / Machine Learning coding interview** task while teaching me how gradient descent and regularization really work under the hood.

---

## 🎯 Objectives
- Implement logistic regression using **gradient descent**.  
- Add **L2 regularization** to prevent overfitting.  
- Handle **numerical stability**, **bias term**, and **class imbalance**.  
- Evaluate performance with accuracy and **Precision–Recall vs Threshold**.

---

## 🧩 Challenge Requirements

### Class to Implement
`LogisticRegressionScratch`  
**Methods:**
- `fit(X, y, lr=0.05, epochs=2000, lambda_=0.0, tol=1e-6, class_weight=None)`
- `predict_proba(X)` → returns probabilities in `[0, 1]`
- `predict(X)` → returns binary labels `{0, 1}`
- *(Stretch)* `predict_threshold(X, t)` → threshold-based prediction

### Functional Details
| Component | Description |
|------------|-------------|
| **Loss Function** | Binary cross-entropy + L2 penalty:  `L = -1/N Σ [y log(p) + (1−y) log(1−p)] + λ/2 * ||w||²` |
| **Optimization** | Batch Gradient Descent |
| **Stability** | Clamp logits `z` in sigmoid between [-30, 30] |
| **Bias Term** | Add a column of ones to `X` |
| **Imbalance Handling** | Weighted loss or threshold tuning |
| **Convergence** | Stop when loss improvement < `tol` or after `epochs` |

---

## 💡 Learning Bite
**Why L2 Regularization?**  
L2 adds a penalty on large weights, helping reduce model variance and preventing weights from exploding when the data is nearly linearly separable.

**Handling Class Imbalance:**  
For skewed datasets, use `class_weight` (e.g., inversely proportional to class frequencies) or adjust the decision threshold to optimize for precision or recall depending on the use case.

---

## 🧠 Example Workflow

```python
from logistic_regression_scratch import LogisticRegressionScratch, accuracy
import numpy as np

# Synthetic data
rng = np.random.default_rng(0)
n0, n1 = 600, 200
X0 = rng.normal([0, 0], 1, (n0, 2))
X1 = rng.normal([2, 2], 1, (n1, 2))
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n0), np.ones(n1)])
idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

# Split
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = LogisticRegressionScratch().fit(X_train, y_train, lr=0.1, epochs=5000, lambda_=0.01)
y_pred = model.predict(X_test)

# Evaluate
print("Test Accuracy:", accuracy(y_test, y_pred))
