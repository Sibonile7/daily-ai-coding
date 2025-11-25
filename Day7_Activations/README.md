# Day 7 — Activation Functions

## 🎯 Objective
Implement core activation functions and their gradients, then test and compare them.  
This challenge prepares you for neural network backpropagation and modern deep-learning architectures.

## Implemented Activations
- ReLU  
- Leaky ReLU  
- Sigmoid (stable)  
- Tanh (stable)  
- Swish  
- GELU (Transformer-style approximation)

## API
```python
activation(x, kind="relu")
activation_grad(x, kind="relu")
