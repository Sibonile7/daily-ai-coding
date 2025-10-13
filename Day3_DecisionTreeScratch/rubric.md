# ✅ Evaluation Rubric — Day 3: Decision Tree Classifier (from Scratch)

## 🎯 Objective
Implement a **binary Decision Tree Classifier** using only **NumPy**, supporting continuous features and Gini impurity.  
Demonstrate understanding of impurity reduction, recursive tree construction, and generalization control.

---

## 🧩 1. Core Implementation (40 pts)

| Criteria | Points | Description |
|:--|:--:|:--|
| **Gini impurity** correctly implemented | 10 | `1 - p₀² - p₁²` computed correctly for any class distribution |
| **Information gain** calculation | 10 | Gain = parent impurity − weighted child impurities |
| **Split selection logic** | 10 | Correctly finds the best (feature, threshold) pair maximizing gain |
| **Leaf node prediction** | 10 | Predicts majority class; handles ties and empty splits gracefully |

---

## ⚙️ 2. Tree Construction Logic (25 pts)

| Criteria | Points | Description |
|:--|:--:|:--|
| **Recursive splitting** | 10 | Properly builds left/right subtrees until stopping criteria |
| **Stopping conditions** | 10 | Uses `max_depth`, `min_samples_split`, and/or `min_impurity_decrease` |
| **Handling continuous features** | 5 | Midpoint thresholds between unique sorted feature values |

---

## 🧠 3. Model Testing & Validation (20 pts)

| Criteria | Points | Description |
|:--|:--:|:--|
| **Train/Test split with randomization** | 5 | 80/20 split with reproducible seed |
| **Baseline comparison** | 5 | Prints model accuracy vs. random or majority baseline |
| **Performance check** | 5 | Achieves ≥85% accuracy on synthetic Gaussian dataset |
| **Experiment reproducibility** | 5 | Code runs without external dependencies |

---

## 💻 4. Code Quality & Structure (10 pts)

| Criteria | Points | Description |
|:--|:--:|:--|
| **Readable, modular code** | 5 | Functions/classes well-organized with clear variable names |
| **Docstrings and comments** | 3 | Each class/function documents its purpose and parameters |
| **No hard-coded data paths** | 2 | Self-contained, portable script |

---

## 🎨 5. Bonus & Extensions (Up to +5 pts)

| Bonus Idea | Points | Description |
|:--|:--:|:--|
| **Entropy criterion option** | +2 | Adds `criterion="entropy"` alternative to Gini |
| **Post-pruning or validation pruning** | +2 | Removes overfitted branches based on validation loss |
| **Decision boundary visualization** | +1 | Matplotlib plot showing class regions |

---

### 💯 Total: **100 pts + up to 5 bonus pts**

---

## 🧾 Submission Checklist
- [ ] `decision_tree_scratch.py` implements the model  
- [ ] `test_decision_tree.py` validates correctness  
- [ ] Code runs with `python test_decision_tree.py`  
- [ ] All key metrics printed clearly (accuracy, depth, etc.)  
- [ ] Report or markdown summary included (`experiment_report.md` or similar)

---

**Evaluator Notes:**  
Use this rubric to self-grade or conduct a peer review.  
A score ≥90% demonstrates strong readiness for ML engineer coding interviews.

