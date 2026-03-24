# Lesson 14 — 2-D Convolution

## Objective

Implement 2-D convolution using both a direct sliding-window kernel and
the im2col + GEMM approach used by deep learning frameworks.

## Prerequisites

- Lesson 11 (Matrix Multiplication), Lesson 05 (2-D Indexing).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Direct convolution | Straightforward but non-coalesced memory access |
| im2col | Unfolds input patches into columns → converts conv to GEMM |
| Valid convolution | No padding; output is (H − KH + 1) × (W − KW + 1) |

## Files

| File | Purpose |
|------|---------|
| `conv2d.cu` | Direct kernel + im2col kernel + CPU reference |
| `conv2d_test.cu` | Parameterised sizes, identity kernel test |

## Build & Run

```bash
cmake --build build --target 14_conv2d
./build/lessons/14_conv2d/14_conv2d
```

## Run Tests

```bash
ctest --test-dir build -R 14_conv2d
```

## What You'll Learn

1. The trade-off between direct convolution and im2col + GEMM.
2. How im2col converts spatial convolution into matrix multiplication.
3. That production libraries (cuDNN, Lesson 19) use autotuned algorithms.
