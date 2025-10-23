# 🧠 Experiment Report — Hard-Margin Linear SVM (NumPy From Scratch)

---

## 📘 Overview
This experiment demonstrates the implementation and performance of a **Hard-Margin Linear Support Vector Machine (SVM)** built entirely from scratch using **NumPy**.  
The objective is to **maximize the separating margin** between two linearly separable classes while maintaining **zero classification errors**.

This challenge is part of the *Daily AI Coding Challenge* series — **Week 1, Day 5** — designed to deepen intuition about classical machine learning optimization methods.

---

## 🎯 Objective
To implement a linear SVM that:
1. **Finds the optimal separating hyperplane** between two linearly separable classes.  
2. **Maximizes the margin** between classes using only NumPy (no libraries like scikit-learn).  
3. **Visualizes** the decision boundary and support vectors in 2D.

---

## 🧩 Experimental Setup

### 🧠 Dataset
- **Type:** Synthetic, Gaussian-distributed clusters  
- **Features:** 2 continuous variables (`x₁`, `x₂`)  
- **Classes:** Binary (−1 and +1)
- **Samples:** 300 total (150 per class)

#### Class Generation
- Positive class: Centered at (2.5, 2.5)
- Negative class: Centered at (−2.5, −2.5)
- Gaussian noise added for slight variation

#### Visualization Example


   + (Positive Class)
    ○ ○ ○ ○ ○ ○ ○
     \       /
      \     /
       \   /
        \ /  ← Decision Boundary
       / \
      /   \
     /     \
    ● ● ● ● ● ● ●
   - (Negative Class)


---

## ⚙️ Model Configuration

| Parameter | Value | Description |
|------------|--------|-------------|
| Learning Rate (`lr`) | 0.05 | Step size for gradient updates |
| Epochs | 20,000 | Maximum number of iterations |
| Regularization (`C`) | 1e5 | Large value → hard-margin constraint |
| Tolerance (`tol`) | 1e-6 | Convergence threshold |
| Random Seed | 0 | Ensures reproducibility |

---

## 🧮 Algorithm Summary

The model solves the primal optimization problem:

\[
\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i (w \cdot x_i + b) \ge 1
\]

- Uses a **large hinge penalty (`C`)** to approximate strict margin constraints.  
- Standardizes features for stable convergence.  
- Stops once all data points satisfy \( y_i (w·x_i + b) ≥ 1 \).

---

## 📊 Results

| Metric | Value | Description |
|---------|--------|-------------|
| **Training Accuracy** | 1.000 | Perfect classification |
| **Misclassifications** | 0 | All samples correctly classified |
| **Margin Width (1/‖w‖)** | Maximized | SVM found optimal separation |
| **Convergence** | Achieved | Early convergence (< 20000 epochs) |

### ✅ Output (Console)

