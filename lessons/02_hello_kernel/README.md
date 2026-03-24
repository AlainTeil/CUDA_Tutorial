# Lesson 02 — Hello Kernel

## Objective

Write and launch your first CUDA kernel. Understand the host/device
execution model, thread indexing, and the GPU memory lifecycle.

## Prerequisites

- Lesson 01 (Device Query).

## Key Concepts

| Concept | Description |
|---------|-------------|
| `__global__` | Declares a kernel callable from the host, executed on the device |
| Thread indexing | `blockIdx.x * blockDim.x + threadIdx.x` gives the global thread ID |
| Memory lifecycle | `cudaMalloc` → `cudaMemcpy` (H→D) → kernel → `cudaMemcpy` (D→H) → `cudaFree` |
| Bounds checking | Threads beyond the array size must exit early |

## Files

| File | Purpose |
|------|---------|
| `hello_kernel.cu` | Threads write their global index to an array |
| `hello_kernel_test.cu` | Parameterised tests across various array sizes |

## Build & Run

```bash
cmake --build build --target 02_hello_kernel
./build/lessons/02_hello_kernel/02_hello_kernel
```

## Run Tests

```bash
ctest --test-dir build -R 02_hello_kernel
```

## What You'll Learn

1. How to define and launch a `__global__` kernel with `<<<grid, block>>>`.
2. The standard thread-indexing formula for 1-D arrays.
3. The full host↔device memory transfer workflow.
4. Why bounds checking is essential when grid size exceeds array size.

## Note

Error checking on CUDA calls is intentionally omitted in this lesson.
Lesson 03 introduces the `CUDA_CHECK` macro to handle errors properly.
