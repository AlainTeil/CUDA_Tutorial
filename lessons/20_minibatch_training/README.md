# Lesson 20 — Mini-Batch Training with cuBLAS

## Objective

Scale training from online (one sample) to mini-batch (B samples packed
into matrices) using cuBLAS GEMM for dramatic efficiency gains.

## Prerequisites

- Lesson 17 (Training), Lesson 19 (cuBLAS).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Mini-batch SGD | Process B samples simultaneously via matrix operations |
| Gradient averaging | Divide accumulated gradients by batch size before update |
| `col_sum` | Parallel column-wise reduction for bias gradients |
| cuBLAS helpers | `gemm_rm`, `gemm_rm_AT` encapsulate the row-major trick |

## Files

| File | Purpose |
|------|---------|
| `minibatch_training.cu` | BatchMLP struct with cuBLAS-backed forward/backward |
| `minibatch_training_test.cu` | Forward, softmax, col_sum, gradient, and convergence tests |

## Build & Run

```bash
cmake --build build --target 20_minibatch_training
./build/lessons/20_minibatch_training/20_minibatch_training
```

## Run Tests

```bash
ctest --test-dir build -R 20_minibatch_training
```

## What You'll Learn

1. Why mini-batch training dramatically improves GPU utilisation.
2. How batch gradients are computed via matrix operations (not per-sample).
3. How to average gradients over the batch dimension.
