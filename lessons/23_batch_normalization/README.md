# Lesson 23 — Batch Normalization

## Objective

Implement Batch Normalization (Ioffe & Szegedy, 2015) — normalise
activations to zero mean / unit variance per mini-batch to stabilise
training.

## Prerequisites

- Lesson 20 (Mini-Batch Training), Lesson 08 (Reduction).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Internal covariate shift | Distribution of layer inputs drifts during training |
| Normalisation | x̂ = (x − μ) / √(σ² + ε) |
| Learnable affine | y = γ · x̂ + β restores representational power |
| Running statistics | EMA of mean/variance for inference mode |
| Backward pass | Chain rule through μ and σ produces a non-trivial dx formula |

## Files

| File | Purpose |
|------|---------|
| `batch_normalization.cu` | Forward (mean, variance, normalise, affine), backward, EMA |
| `batch_normalization_test.cu` | Zero-mean/unit-var, affine scaling, inference mode, gradient checks |

## Build & Run

```bash
cmake --build build --target 23_batch_normalization
./build/lessons/23_batch_normalization/23_batch_normalization
```

## Run Tests

```bash
ctest --test-dir build -R 23_batch_normalization
```

## What You'll Learn

1. Why normalising layer inputs accelerates and stabilises training.
2. The difference between training mode (batch stats) and inference mode (running stats).
3. How shared-memory reductions compute per-channel mean and variance.
4. The full backward pass through the normalisation formula.
