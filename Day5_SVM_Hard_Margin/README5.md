# 🧠 Day 5 — Hard-Margin Linear SVM (NumPy From Scratch)

### Overview
This project implements a **Hard-Margin Linear Support Vector Machine (SVM)** entirely from scratch using **NumPy**, without relying on scikit-learn or TensorFlow.  
It is part of the **Daily AI Coding Challenge Series** (Week 1, Day 5) and is designed to strengthen understanding of **margin maximization**, **geometric intuition**, and **optimization** in classical machine learning.

The goal is to find a **hyperplane** that perfectly separates two linearly separable classes while **maximizing the margin** between them.

---

## Project Structure

| File | Description |
|------|--------------|
| `svm_hard_margin.py` | Core implementation of the NumPy-based Hard-Margin SVM with 2D visualization. |
| `test_svm_hard_margin.py` | Sanity check — validates the implementation on synthetic separable data. |
| `experiment_report.md` | Technical summary and interpretation of experiment results. |
| `README.md` | Project documentation (this file). |

---

## ⚙️ Mathematical Foundation

We solve the **hard-margin SVM optimization problem**:

\[
\min_{w,b} \frac{1}{2} \|w\|^2
\]

subject to:

\[
y_i (w \cdot x_i + b) \geq 1, \quad \forall i
\]

where:
- \( w \) → weight vector  
- \( b \) → bias term  
- \( y_i \in \{-1, +1\} \) → class labels  
- \( x_i \) → feature vector

The optimization seeks the **maximum-margin hyperplane**, i.e., the one with the largest distance to the nearest points (support vectors).

---

## Implementation Details

### 🔧 Core Features
- **NumPy-only** (no ML libraries)
- **Gradient-based optimization** with a large hinge penalty (approximating hard constraints)
- **Feature standardization** for stable training
- **Convergence detection** when all constraints are satisfied
- **2D visualization** of the decision boundary and margins

### Algorithm Steps
1. Initialize weights and bias randomly  
2. Standardize features  
3. Compute margins: \( m_i = y_i (w \cdot x_i + b) \)  
4. Identify violations where \( m_i < 1 \)  
5. Update weights:
   \[
   w \leftarrow w - \eta \left( w - C \sum_{i: m_i<1} y_i x_i \right)
   \]
6. Stop when all samples satisfy \( m_i \geq 1 \)

---

## Usage Instructions

### 1️⃣ Install Dependencies
```bash
pip install numpy matplotlib
