# 🧠 Daily AI Coding Challenge — Day 8  
## **Perceptron Learning Algorithm (PLA)**  

### 🎯 Objective  
Implement a **Perceptron classifier** from scratch using NumPy.  
The model should:
- Train on 2D linearly separable data.
- Update weights using the PLA rule.  
- Visualize decision boundary evolution.
- Report the number of updates until convergence.

### Key Concepts
- **Linear classification** and **online learning**.  
- **PLA** as the foundation of neural networks.  
- Why convergence occurs **only for linearly separable data**.

### Learning Bite
The perceptron minimizes classification errors by adjusting weights whenever a misclassification occurs:
\[
w := w + \eta (y_i - \hat{y_i}) x_i
\]
It converges only if a perfect hyperplane exists.

### 💡 Bonus Idea
- Add a learning rate schedule (`η_t = η / (1 + decay * t)`).
- Try non-separable data and observe that it fails to converge.

### Example Output
