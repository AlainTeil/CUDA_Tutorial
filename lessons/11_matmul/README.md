# Lesson 11 — Matrix Multiplication

## Objective

Implement matrix multiplication (GEMM) — the foundational operation for
neural networks — in both a naive and a tiled shared-memory version.

## Prerequisites

- Lesson 04 (Shared Memory), Lesson 10 (Transpose / Tiling).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Arithmetic intensity | FLOP / byte — determines whether a kernel is memory- or compute-bound |
| Naive matmul | Each thread reads one full row & column from global memory |
| Tiled matmul | Shared-memory tiles reduce global reads by factor TILE |
| CPU reference | Essential for validating GPU results |

## Files

| File | Purpose |
|------|---------|
| `matmul.cu` | Naive and tiled matmul kernels + CPU baseline |
| `matmul_test.cu` | Identity matrix, various dimensions, tolerance checks |

## Build & Run

```bash
cmake --build build --target 11_matmul
./build/lessons/11_matmul/11_matmul
```

## Run Tests

```bash
ctest --test-dir build -R 11_matmul
```

## What You'll Learn

1. Why matrix multiplication is compute-bound (high arithmetic intensity).
2. How tiling reduces global memory traffic by a factor of TILE.
3. The importance of `__syncthreads()` between shared-memory load/compute.
4. That production code should use cuBLAS (Lesson 19).
