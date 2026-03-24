# Lesson 29 — Residual Connections & Layer Normalization

## Objective

Implement the two structural components that enable deep Transformers:
residual (skip) connections and layer normalisation.

## Prerequisites

- Lesson 23 (Batch Normalization concepts), Lesson 08 (Reduction).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Residual connection | y = x + F(x); prevents gradient vanishing in deep networks |
| Layer normalisation | Normalises per-sample (not per-batch); suits variable-length sequences |
| Welford's algorithm | Numerically stable online computation of mean and variance |
| Fused kernel | Residual add + LayerNorm in one kernel saves a global memory round-trip |

## Parts

- **Part 1** — Residual (skip) connection.
- **Part 2** — LayerNorm forward (Welford's online algorithm).
- **Part 3** — LayerNorm backward (dx, dγ, dβ).
- **Part 4** — Fused residual + LayerNorm kernel.

## Files

| File | Purpose |
|------|---------|
| `residual_layernorm.cu` | All four parts |
| `residual_layernorm_test.cu` | Zero-mean/unit-var, finite-difference gradients, fused vs separate |

## Build & Run

```bash
cmake --build build --target 29_residual_layernorm
./build/lessons/29_residual_layernorm/29_residual_layernorm
```

## Run Tests

```bash
ctest --test-dir build -R 29_residual_layernorm
```

## What You'll Learn

1. Why residual connections are essential for training deep networks.
2. How LayerNorm differs from BatchNorm (per-sample vs per-batch).
3. Why Welford's algorithm is preferred for numerical stability.
4. How kernel fusion reduces global memory traffic.
