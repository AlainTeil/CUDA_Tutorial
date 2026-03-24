# Lesson 10 — Matrix Transpose

## Objective

Demonstrate the memory coalescing challenge in matrix transpose and solve
it with shared-memory tiling and bank-conflict-free padding.

## Prerequisites

- Lesson 04 (Shared Memory), Lesson 05 (2-D Indexing).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Naive transpose | Coalesced reads, scattered writes — ~50 % peak bandwidth |
| Tiled transpose | Shared memory tile enables coalesced reads AND writes |
| Bank conflicts | 32 banks; same-bank access serialises threads in a warp |
| "+1" padding | Declaring `tile[32][33]` eliminates bank conflicts |

## Files

| File | Purpose |
|------|---------|
| `transpose.cu` | Naive and tiled transpose kernels |
| `transpose_test.cu` | CPU reference, double transpose is identity |

## Build & Run

```bash
cmake --build build --target 10_transpose
./build/lessons/10_transpose/10_transpose
```

## Run Tests

```bash
ctest --test-dir build -R 10_transpose
```

## What You'll Learn

1. Why naive transpose wastes half the available memory bandwidth.
2. How shared-memory tiling solves the write-coalescing problem.
3. What bank conflicts are and how "+1" padding eliminates them.
