
# Neural Logic Gates from Scratch in C

A lightweight, zero-dependency machine learning engine built entirely from scratch in pure C. This repository demonstrates the low-level mathematical fundamentals of artificial neural networks—specifically tracking the evolution from a single-layer **Perceptron** to a **Multi-Layer Perceptron (MLP)**—using digital logic gates as a computational benchmark.



---

## 🚀 The Core Philosophy

Most modern AI practitioners rely heavily on abstraction layers like PyTorch, TensorFlow, or NumPy (`model.fit()`). This project strips away those abstractions to expose the raw mechanics underneath:
* **No external libraries** (only standard native headers: `<stdio.h>`, `<math.h>`, `<stdlib.h>`).
* **Manual memory layout** utilizing standard C native arrays instead of heavy tensor structures.
* **Pure mathematical implementation** of forward propagation, backpropagation, and gradient descent.

---

## 🧠 The Architectural Evolution

This repository tracks a famous milestone in AI history: **The AI Winter of 1969**, where Marvin Minsky proved that single-layer neural networks cannot solve non-linearly separable problems like the XOR gate.

### 1. Single-Layer Perceptron (`AND` / `OR` Gates)
For linearly separable functions, a single neural node acts as a "mathematical knife," drawing a single straight decision boundary line to split `0` and `1` outputs. 
* **`AND` Target Matrix:** `{0, 0, 0, 1}`
* **`OR` Target Matrix:** `{0, 1, 1, 1}`

### 2. Multi-Layer Perceptron (`XOR` Gate)
Because the XOR gate contains diagonal outputs (`{0, 1, 1, 0}`), it is geometrically non-linear. To solve this, the architecture is upgraded to include a **Hidden Layer** with 2 hidden neurons. This effectively gives the network two distinct decision boundaries, warping the 2D space so the data becomes linearly separable for the final output node.



---

## 📐 The Mathematical Engine

Every training iteration passes data through a three-step algorithmic pipeline:

1. **Forward Propagation:** Inputs ($x_n$) are multiplied by their respective weights ($w_n$), combined with a bias ($b$), and squashed through a custom **Sigmoid Activation Function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

2. **Backpropagation:** The network calculates the prediction error at the output layer and uses the **Calculus Chain Rule** to propagate gradients backward through the hidden layers:
   $$\frac{\partial \text{Error}}{\partial w} = \frac{\partial \text{Error}}{\partial \text{Output}} \times \frac{\partial \text{Output}}{\partial \text{Sum}} \times \frac{\partial \text{Sum}}{\partial w}$$

3. **Gradient Descent Optimization:** Internal parameters are iteratively shifted closer to the global minimum using the calculated adjustments scaled by a continuous `learning_rate` ($\eta$):
   $$w \leftarrow w + (\eta \cdot \delta \cdot \text{input})$$

---

## 🛠️ Compilation and Execution

Since this engine is written in pure, standard C, you can compile and run it instantly on any operating system using `gcc`.

### To Run the Multi-Layer XOR Network:
```bash
# Compile the program linking the math library (-lm is required for <math.h> exp() on Linux/macOS)
gcc xor_gate.c -o xor_gate -lm

# Execute the binary
./xor_gate

```

### Expected Output Structure:

The network initializes with minor pseudo-random weights, resulting in chaotic guesses early on. After iterating through its epochs, the system converges perfectly to the global minimum:

```text
--- Training XOR Network Starting ---
--- Training Complete ---

Testing the XOR network:
Input: 0, 0 -> Predicted Probability: 0.0124 (Rounded: 0)
Input: 0, 1 -> Predicted Probability: 0.9841 (Rounded: 1)
Input: 1, 0 -> Predicted Probability: 0.9843 (Rounded: 1)
Input: 1, 1 -> Predicted Probability: 0.0152 (Rounded: 0)

```

*(Note: The raw decimal values represent the network's calculated probability/confidence spectrum prior to hitting the 50% threshold `round()` filter).*

---

## 📈 Real-World Applications of Low-Level C Networks

While logic gates are standard conceptual benchmarks, writing neural models natively in C is a crucial industry pattern for:

* **TinyML / Edge Computing:** Deploying highly performant, low-footprint predictive intelligence on power-constrained microcontrollers (e.g., smart sensors, medical pacemakers).
* **Noise-Tolerant Hardware Scanners:** Creating systems that evaluate real-world electrical circuit components without brittle `if/else` limits.

```

