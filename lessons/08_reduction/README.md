# Lesson 08 — Parallel Reduction

## Objective

Implement the "many-to-one" parallel reduction primitive using two
approaches: shared-memory tree reduction and warp-shuffle reduction.

## Prerequisites

- Lesson 04 (Shared Memory), Lesson 05 (Thread Hierarchy).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Tree reduction | Sequential addressing with stride halving avoids warp divergence |
| Warp shuffle | `__shfl_down_sync` exchanges data via registers (no shared memory) |
| Multi-block reduction | Recursive driver reduces per-block partial sums to a scalar |
| Work efficiency | O(n) operations for n elements |

## Files

| File | Purpose |
|------|---------|
| `reduction.cu` | Shared-memory and warp-shuffle reduction kernels |
| `reduction_test.cu` | Parameterised tests (1 to 2²⁰ elements) |

## Build & Run

```bash
cmake --build build --target 08_reduction
./build/lessons/08_reduction/08_reduction
```

## Run Tests

```bash
ctest --test-dir build -R 08_reduction
```

## What You'll Learn

1. Why sequential addressing is better than interleaved for reductions.
2. How warp-level `__shfl_down_sync` eliminates shared-memory overhead.
3. How to recursively reduce across multiple blocks.
4. That production code should prefer CUB / Thrust for reductions.
