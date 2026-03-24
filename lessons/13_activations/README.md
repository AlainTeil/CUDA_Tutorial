# Lesson 13 — Activation Functions

## Objective

Implement element-wise (ReLU, Sigmoid, Tanh) and row-wise (Softmax)
activation functions with their backward passes.

## Prerequisites

- Lesson 12 (Dense Layer).

## Key Concepts

| Concept | Description |
|---------|-------------|
| ReLU | max(0, x) — avoids saturation; gradient is the step function |
| Sigmoid | 1 / (1 + e⁻ˣ) — outputs in (0, 1); saturates for large |x| |
| Tanh | tanh(x) — outputs in (-1, 1); derivative 1 − tanh²(x) |
| Softmax | Row-wise normalisation; uses max-subtraction for numerical stability |

## Files

| File | Purpose |
|------|---------|
| `activations.cu` | Forward and backward kernels for all four activations |
| `activations_test.cu` | Range checks, sum-to-one, finite-difference gradients |

## Build & Run

```bash
cmake --build build --target 13_activations
./build/lessons/13_activations/13_activations
```

## Run Tests

```bash
ctest --test-dir build -R 13_activations
```

## What You'll Learn

1. Why ReLU is preferred over sigmoid/tanh (no saturation).
2. The max-subtraction trick that makes softmax numerically stable.
3. How shared-memory reductions implement row-wise softmax on the GPU.
