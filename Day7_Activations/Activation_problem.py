"""
Day 7 — Activation Functions
----------------------------

Implement:
- activation(x, kind)
- activation_grad(x, kind)
- support: relu, leaky_relu, sigmoid, tanh, swish, gelu
- stable sigmoid & tanh
- optional MLP demo for activation comparison
"""

import numpy as np


# ---------------- Activation Functions ----------------
def activation(x: np.ndarray, kind: str = "relu", alpha: float = 0.01):
    x = np.asarray(x, dtype=float)

    if kind == "relu":
        return np.maximum(0, x)

    elif kind == "leaky_relu":
        return np.where(x > 0, x, alpha * x)

    elif kind == "sigmoid":
        # stable sigmoid
        x_clip = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clip))

    elif kind == "tanh":
        x_clip = np.clip(x, -20, 20)
        return np.tanh(x_clip)

    elif kind == "swish":
        return x * activation(x, "sigmoid")

    elif kind == "gelu":
        # approximate gelu
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    else:
        raise ValueError(f"Unknown activation: {kind}")


# ---------------- Activation Gradients ----------------
def activation_grad(x: np.ndarray, kind: str = "relu", alpha: float = 0.01):
    x = np.asarray(x, dtype=float)

    if kind == "relu":
        return (x > 0).astype(float)

    elif kind == "leaky_relu":
        grad = np.ones_like(x)
        grad[x < 0] = alpha
        return grad

    elif kind == "sigmoid":
        s = activation(x, "sigmoid")
        return s * (s - 1) * -1  # s*(1-s)

    elif kind == "tanh":
        t = activation(x, "tanh")
        return 1 - t**2

    elif kind == "swish":
        s = activation(x, "sigmoid")
        return s + x * s * (1 - s)

    elif kind == "gelu":
        # derivative of approximate gelu
        x3 = x**3
        term = np.sqrt(2/np.pi) * (x + 0.044715 * x3)
        tanh_term = np.tanh(term)
        sech2_term = 1 - tanh_term**2
        return 0.5 * (1 + tanh_term) + \
               0.5 * x * sech2_term * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)

    else:
        raise ValueError(f"Unknown activation: {kind}")


# ---------------- Optional Mini-MLP Demo ----------------
def mlp_forward(X, W1, b1, W2, b2, act="relu"):
    Z1 = activation(X @ W1 + b1, act)
    Z2 = Z1 @ W2 + b2
    return Z2  # raw output (logits)


def mlp_predict(X, W1, b1, W2, b2, act="relu"):
    logits = mlp_forward(X, W1, b1, W2, b2, act)
    return (logits > 0).astype(int)


if __name__ == "__main__":
    print("Day 7 Activations module loaded successfully.")
