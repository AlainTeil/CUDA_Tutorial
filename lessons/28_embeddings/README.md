# Lesson 28 — Token Embeddings & Positional Encoding

## Objective

Implement the input stage of Transformer models: token embedding lookup,
sinusoidal positional encoding, and embedding backward with gradient
scatter.

## Prerequisites

- Lesson 19 (cuBLAS), Lesson 12 (Dense Layer backward).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Token embedding | Gather rows from a (V × D) table by token ID |
| Sinusoidal PE | Fixed frequency encoding: sin/cos at exponentially-spaced wavelengths |
| Embedding backward | `atomicAdd` scatter because duplicate token IDs share gradient |
| Output projection | cuBLAS GEMM for final linear transformation |

## Parts

- **Part 1** — Token embedding lookup (gather).
- **Part 2** — Sinusoidal positional encoding.
- **Part 3** — Embedding backward (atomicAdd scatter).
- **Part 4** — cuBLAS matmul for output projection.

## Files

| File | Purpose |
|------|---------|
| `embeddings.cu` | All four parts + main demo |
| `embeddings_test.cu` | Row-copy, PE properties, gradient accumulation, projection tests |

## Build & Run

```bash
cmake --build build --target 28_embeddings
./build/lessons/28_embeddings/28_embeddings
```

## Run Tests

```bash
ctest --test-dir build -R 28_embeddings
```

## What You'll Learn

1. How embedding lookup is a pure gather operation (no computation).
2. Why positional encoding uses exponentially-spaced frequencies.
3. Why `atomicAdd` is needed when multiple tokens share the same ID.
