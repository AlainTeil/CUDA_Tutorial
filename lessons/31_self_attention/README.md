# Lesson 31 — Multi-Head Self-Attention

## Objective

Implement the core mechanism of Transformers: multi-head scaled
dot-product attention with batched cuBLAS GEMMs.

## Prerequisites

- Lesson 19 (cuBLAS), Lesson 28 (Embeddings), Lesson 13 (Softmax).

## Key Concepts

| Concept | Description |
|---------|-------------|
| QKV projection | Single GEMM produces Q, K, V from input |
| Split heads | Reshape (B, T, D) → (B, h, T, d_k) |
| Scaled dot-product | scores = Q·Kᵀ / √d_k; weights = softmax(scores); context = weights · V |
| Merge heads | Reshape (B, h, T, d_k) → (B, T, D) |
| Output projection | Final linear transformation via GEMM |

## Files

| File | Purpose |
|------|---------|
| `self_attention.cu` | All five attention components + main demo |
| `self_attention_test.cu` | Split/merge round-trip, softmax properties, identity weight tests |

## Build & Run

```bash
cmake --build build --target 31_self_attention
./build/lessons/31_self_attention/31_self_attention
```

## Run Tests

```bash
ctest --test-dir build -R 31_self_attention
```

## What You'll Learn

1. How multi-head attention processes multiple representation subspaces.
2. Why 1/√d_k scaling prevents softmax saturation for large dimensions.
3. How batched `cublasSgemmStridedBatched` efficiently handles all heads.
4. How split/merge reshaping enables per-head independent computation.
