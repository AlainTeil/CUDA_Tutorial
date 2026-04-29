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
| im2col + tiled GEMM | Full pipeline: `im2col_kernel` → 16 × 16 tiled matmul → output |
| Backward (dkernel) | Cross-correlation of upstream gradient with input |
| Backward (dx) | Scatter-with-atomics from upstream gradient through rotated kernel |
| Valid convolution | No padding; output is (H − KH + 1) × (W − KW + 1) |

## Files

| File | Purpose |
|------|---------|
| `conv2d.cu` | Forward (direct + im2col+tiled-GEMM) **and** backward (`conv2d_backward_dkernel`, `conv2d_backward_dx`) + CPU reference |
| `conv2d_test.cu` | Direct + im2col→GEMM forward tests over five (H, W, KH, KW) shapes; identity-kernel sanity check; finite-difference checks for `dkernel` and `dx` |

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
3. How an inlined tiled GEMM (the same pattern as Lesson 11) finishes the job once the columns are laid out.
4. How the backward pass derives `dkernel` (cross-correlation `dout ⋆ in`) and `dx` (scatter through the rotated kernel) from the same forward equation.
5. That production libraries (cuDNN, Lesson 19) use autotuned algorithms.
