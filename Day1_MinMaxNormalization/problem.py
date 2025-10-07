"""
Day 1 — Min–Max Normalization
"""
import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    """Scale the array to [0, 1] using min–max normalization.

    Handles NaNs, constant columns, and preserves shape.
    """
    if x.size == 0:
        return np.array([], dtype=float).reshape(x.shape)

    x = x.astype(float)
    min_ = np.nanmin(x, axis=0, keepdims=True)
    max_ = np.nanmax(x, axis=0, keepdims=True)
    scale = np.where(max_ == min_, 1, max_ - min_)

    normalized = (x - min_) / scale
    normalized = np.where(max_ == min_, 0, normalized)
    return normalized


class Normalizer:
    """Reusable min–max normalizer with fit/transform/inverse_transform."""
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = np.nanmin(x, axis=0, keepdims=True)
        self.max_ = np.nanmax(x, axis=0, keepdims=True)
        return self

    def transform(self, x: np.ndarray):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Call fit() before transform().")
        scale = np.where(self.max_ == self.min_, 1, self.max_ - self.min_)
        normalized = (x - self.min_) / scale
        normalized = np.where(self.max_ == self.min_, 0, normalized)
        return normalized

    def inverse_transform(self, x_norm: np.ndarray):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Call fit() before inverse_transform().")
        scale = np.where(self.max_ == self.min_, 1, self.max_ - self.min_)
        return x_norm * scale + self.min_
