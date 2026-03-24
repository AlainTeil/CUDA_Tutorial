# Lesson 12 — Dense (Fully Connected) Layer

## Objective

Implement the forward and backward passes of a dense layer — the first
trainable building block of a neural network.

## Prerequisites

- Lesson 11 (Matrix Multiplication).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Forward | Y = X · W + b |
| Backward (dX) | dX = dY · Wᵀ |
| Backward (dW) | dW = Xᵀ · dY |
| Backward (db) | db = column-wise sum of dY |
| Gradient checking | Finite-difference verification of analytical gradients |

## Files

| File | Purpose |
|------|---------|
| `dense_layer.cu` | Forward and backward with GPU matmul + transpose |
| `dense_layer_test.cu` | Forward correctness + finite-difference gradient checks |

## Build & Run

```bash
cmake --build build --target 12_dense_layer
./build/lessons/12_dense_layer/12_dense_layer
```

## Run Tests

```bash
ctest --test-dir build -R 12_dense_layer
```

## What You'll Learn

1. How the chain rule produces three gradient terms (dX, dW, db).
2. Why transpose operations are needed in the backward pass.
3. How to validate gradients numerically with finite differences.
