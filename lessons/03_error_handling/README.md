# Lesson 03 — Error Handling

## Objective

Introduce robust error handling for all CUDA API calls and kernel launches.
From this lesson onward, every CUDA call in the tutorial is wrapped in
`CUDA_CHECK`.

## Prerequisites

- Lesson 02 (Hello Kernel).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Synchronous errors | Returned immediately by CUDA API calls (`cudaMalloc`, `cudaMemcpy`, …) |
| Asynchronous errors | Detected later via `cudaGetLastError()` after kernel launches |
| `CUDA_CHECK` macro | Checks return codes and aborts with file/line info on failure |
| `CUDA_CHECK_THROW` | Exception-based variant for use in test frameworks |

## Files

| File | Purpose |
|------|---------|
| `error_handling.cu` | Defines both error-handling macros and demonstrates usage |
| `error_handling_test.cu` | Tests valid calls, invalid calls, kernel errors, and message quality |

## Build & Run

```bash
cmake --build build --target 03_error_handling
./build/lessons/03_error_handling/03_error_handling
```

## Run Tests

```bash
ctest --test-dir build -R 03_error_handling
```

## What You'll Learn

1. The difference between synchronous and asynchronous CUDA errors.
2. How to define a reusable `CUDA_CHECK` macro with `do { … } while (0)`.
3. When to use `cudaGetLastError()` vs `cudaPeekAtLastError()`.
4. How to reset the CUDA error state after intentional failures in tests.
