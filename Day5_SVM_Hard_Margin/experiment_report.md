### 🧾 **experiment_report.md**
```markdown
# Hard-Margin Linear SVM — Experiment Report

## Setup
- **Goal:** Train a NumPy-only hard-margin linear SVM and visualize the separator + margins.
- **Data:** Synthetic Gaussian clusters (clearly separable).

## Plot
Solid line = decision boundary  
Dashed lines = margin boundaries (distance = 1/‖w‖)

## Interpretation
- Hard margin = zero tolerance for misclassification.
- Maximizes distance between classes → wider margin, better generalization.
- If data not separable → must use soft margin (smaller `C`).
