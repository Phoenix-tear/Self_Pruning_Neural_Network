# Self-Pruning Neural Network with Learnable Gates

## 1. Overview

This project implements a CIFAR-10 classifier that **learns to prune itself during training** using learnable gates on each weight.

* Each weight has a gate ( g \in (0,1) )
* Gates are optimized via gradient descent
* An **L1 penalty** encourages sparsity
* Less important connections → gates → 0 (pruned)

---

## 2. Method

### 2.1 PrunableLinear Layer

Standard linear layer:
[
y = Wx + b
]

With gates:
[
g_{ij} = \sigma(s_{ij}) = \frac{1}{1 + e^{-s_{ij}}}
]

Effective weight:
[
w_{ij}^{'} = w_{ij} \cdot g_{ij}
]

Layer output:
[
y_k = \sum_{j=1}^{d_{in}} (w_{kj} \cdot g_{kj}) x_j + b_k
]

Implementation:

* `gates = sigmoid(gate_scores)`
* `pruned_weight = weight * gates`
* Fully differentiable → trained with Adam

---

### 2.2 Loss Function

[
L = L_{cls} + \lambda L_{sp}
]

Where:

* ( L_{cls} ): Cross-entropy loss
* ( L_{sp} = \sum g_{ij} ): L1 penalty on gates

Effect:

* Larger ( \lambda ) → more sparsity
* Smaller ( \lambda ) → better accuracy

---

## 3. Why L1 Causes Sparsity

### L1 vs L2

* **L1** → pushes values to exactly 0 → sparse
* **L2** → shrinks values but rarely zero

### In Gates

* Unimportant weights → ( g_{ij} \to 0 )
* Important weights → stay high

Result:

* Near-binary behavior (0 = pruned, 1 = active)

---

## 4. Architecture & Training

### Model

* Input: 3072 (flattened CIFAR-10 image)
* Hidden 1: 3072 → 1024 (ReLU)
* Hidden 2: 1024 → 512 (ReLU)
* Output: 512 → 10

All layers use **PrunableLinear**

### Training Setup

* Dataset: CIFAR-10
* Optimizer: Adam (lr = ( 1 \times 10^{-3} ))
* Epochs: 30
* Device: GPU (if available)

---

## 5. Results

| λ    | Accuracy (%) | Sparsity (%) |
| ---- | ------------ | ------------ |
| 1e-5 | 58.90        | 50.99        |
| 1e-4 | 59.08        | 91.76        |
| 1e-3 | 51.57        | 99.80        |

### Observations

* **( \lambda = 1e{-4} )** → best trade-off
* High sparsity with minimal accuracy loss
* Too large ( \lambda ) → performance drops

---

## 6. Gate Distribution

* Many gates near **0 → pruned**
* Some gates near **1 → active**
* Clear **bimodal distribution**

---

## 7. Key Takeaways

* Learns structure **during training** (no post-pruning)
* L1 gating provides **automatic model compression**
* ( \lambda ) controls **accuracy vs sparsity trade-off**
* Produces **efficient, compact networks**

---

## Conclusion

L1-penalized gating is a simple and effective method to train **highly sparse neural networks** with competitive performance on CIFAR-10.
