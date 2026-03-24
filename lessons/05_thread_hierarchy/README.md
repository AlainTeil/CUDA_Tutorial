# Lesson 05 — Thread Hierarchy

## Objective

Understand multidimensional grid and block organisation using `dim3`.
Learn how to compute global indices for 1-D, 2-D, and 3-D problems.

## Prerequisites

- Lessons 01–03.

## Key Concepts

| Concept | Description |
|---------|-------------|
| `dim3` | Specifies grid / block dimensions in up to 3 dimensions |
| 1-D indexing | `idx = blockIdx.x * blockDim.x + threadIdx.x` |
| 2-D indexing | Row-major: `row * cols + col` |
| 3-D indexing | `z * (rows * cols) + row * cols + col` |

## Files

| File | Purpose |
|------|---------|
| `thread_hierarchy.cu` | Three kernels: `fill_1d`, `fill_2d`, `fill_3d` |
| `thread_hierarchy_test.cu` | Parameterised tests across various dimensions |

## Build & Run

```bash
cmake --build build --target 05_thread_hierarchy
./build/lessons/05_thread_hierarchy/05_thread_hierarchy
```

## Run Tests

```bash
ctest --test-dir build -R 05_thread_hierarchy
```

## What You'll Learn

1. How to specify multidimensional grids and blocks with `dim3`.
2. Global index formulas for 1-D, 2-D, and 3-D arrays.
3. Why multidimensional indexing matters for images, matrices, and volumes.
