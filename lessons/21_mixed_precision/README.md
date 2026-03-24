# Lesson 21 — Mixed-Precision Training

## Objective

Exploit Tensor Cores via FP16 and TF32 precision while maintaining FP32
master weights for numerical stability — the standard approach for modern
deep learning training.

## Prerequisites

- Lesson 20 (Mini-Batch + cuBLAS).

## Key Concepts

| Concept | Description |
|---------|-------------|
| FP32 baseline | Full precision; correct but slower on Tensor Core hardware |
| TF32 mode | Transparent on Ampere+; same code, ~2× faster via `CUBLAS_TF32_TENSOR_OP_MATH` |
| FP16 mode | Explicit half-precision via `__half`; requires loss scaling |
| Loss scaling | Scale loss to prevent gradient underflow in FP16 range |
| Master weights | FP32 copy updated with unscaled FP32 gradients |

## Files

| File | Purpose |
|------|---------|
| `mixed_precision.cu` | Three-mode MLP training (FP32, TF32, FP16+loss scaling) |
| `mixed_precision_test.cu` | Conversion accuracy, convergence, and mode comparison tests |

## Build & Run

```bash
cmake --build build --target 21_mixed_precision
./build/lessons/21_mixed_precision/21_mixed_precision
```

## Run Tests

```bash
ctest --test-dir build -R 21_mixed_precision
```

## What You'll Learn

1. How FP16 and TF32 exploit Tensor Cores for faster GEMM.
2. Why loss scaling prevents gradient underflow in half precision.
3. The dual-storage pattern: FP16 for compute, FP32 for accumulation.
