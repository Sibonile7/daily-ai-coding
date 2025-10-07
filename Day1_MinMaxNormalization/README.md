# 🧠 Day 1 — Min–Max Normalization (Interview + Learning)

## 🎯 Prompt
Implement `normalize(x)` that performs min–max scaling to [0, 1] for a NumPy array.
- Accept 1D or 2D numeric input.
- Return float output; preserve shape.
- Handle constant columns (all same value) by returning zeros for that column.
- Be tolerant to NaNs using `np.nanmin` / `np.nanmax`.
- Edge cases: empty array, NaNs, constant column, mixed dtypes.

**Stretch:** add a `Normalizer` class with `fit`, `transform`, `inverse_transform`.

## 🧠 Why this matters
Scaling helps gradient-based optimization and any distance-based model (k-NN, SVMs, clustering).

## ✅ Rubric
- Correctness on edge cases — 40%
- Vectorization / efficiency — 25%
- Clean API & docstring — 20%
- Tests for edge cases — 15%
