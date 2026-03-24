# Lesson 25 — Optimizers & Learning-Rate Schedules

## Objective

Compare SGD, SGD+Momentum, Adam, and AdamW optimisers on the Rosenbrock
function. Demonstrate warmup-cosine learning-rate scheduling integrated
into training.

## Prerequisites

- Lesson 17 (Training Loop).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Vanilla SGD | Global learning rate; no memory of past gradients |
| Momentum | Exponential moving average of gradients accelerates convergence |
| Adam | Adaptive per-parameter rates via first & second moment tracking |
| Bias correction | Unbias early moment estimates: m̂ = m / (1 − β₁ᵗ) |
| AdamW | Decoupled weight decay applied before gradient step |
| Warmup-cosine LR | Linear ramp-up followed by cosine annealing |

## Files

| File | Purpose |
|------|---------|
| `optimizers.cu` | SGD, momentum, Adam, AdamW kernels; LR schedulers; Rosenbrock demo |
| `optimizers_test.cu` | Convergence, weight decay, LR schedule formula verification |

## Build & Run

```bash
cmake --build build --target 25_optimizers
./build/lessons/25_optimizers/25_optimizers
```

## Run Tests

```bash
ctest --test-dir build -R 25_optimizers
```

## What You'll Learn

1. Why Adam converges faster than SGD on ill-conditioned landscapes.
2. How bias correction prevents early-training divergence.
3. How AdamW separates weight decay from the adaptive gradient step.
4. How warmup-cosine schedules modulate the learning rate during training.
