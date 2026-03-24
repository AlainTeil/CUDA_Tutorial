# Lesson 04 — Memory Model

## Objective

Explore the CUDA memory hierarchy: global, constant, and shared memory.
Understand coalescing, broadcast caching, and synchronisation with
`__syncthreads()`.

## Prerequisites

- Lessons 01–03.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Global memory | Large, high-latency; coalesced access is critical for bandwidth |
| Constant memory | Read-only; broadcasts a single value to all threads in a warp |
| Shared memory | Fast, per-block; requires `__syncthreads()` for consistency |
| Halo cells | Extra shared-memory elements for stencil boundary handling |

## Parts

- **Part A** — Vector add (global memory, coalescing).
- **Part B** — Polynomial evaluation (constant memory).
- **Part C** — 1-D stencil with halo cells (shared memory).

## Files

| File | Purpose |
|------|---------|
| `memory_model.cu` | Three kernels demonstrating each memory type |
| `memory_model_test.cu` | Parameterised correctness tests for all three parts |

## Build & Run

```bash
cmake --build build --target 04_memory_model
./build/lessons/04_memory_model/04_memory_model
```

## Run Tests

```bash
ctest --test-dir build -R 04_memory_model
```

## What You'll Learn

1. How coalesced global memory access maximises bandwidth.
2. How `__constant__` memory and `cudaMemcpyToSymbol` work.
3. How shared memory reduces global reads via data reuse.
4. Proper placement of `__syncthreads()` between load and compute phases.
