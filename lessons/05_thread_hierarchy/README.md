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

## A Note on Block Size — Why 256 Is the Common Default

Throughout the rest of this tutorial you will see `constexpr int kBlockSize = 256`
appear in nearly every lesson. The choice is deliberate but not magical:

- **Multiple of 32 (warp size).** Block size *must* be a multiple of 32 on every
  current NVIDIA GPU; otherwise the trailing threads in the last warp run as
  inactive lanes and waste hardware.
- **Power of two.** Powers of two simplify tree reductions and binary
  shared-memory layouts (used in lessons 8, 9, 23, …).
- **Sweet spot for occupancy.** On modern compute capabilities (≥ sm_75) the SM
  can hold many resident blocks. 256 threads/block typically allows 4–8 blocks
  per SM under realistic register/shared-memory budgets, hitting peak
  warp-level occupancy without starving for blocks.
- **Headroom for shared memory.** A larger block (e.g. 1024) often forces
  occupancy below 50 % once you add per-thread shared memory; a smaller block
  (e.g. 64) leaves shared-memory bandwidth on the table.

Other common values you will encounter in this tutorial:
- **128** — used when each thread does heavy register work (e.g. tiled GEMM).
- **`kTile = 16`** in lessons 11 and 14 — the side length of a 16 × 16 = 256
  thread tile, again so that each block is a full 8 warps.
- **`betas = (0.9, 0.999)` and `eps = 1e-8`** in lesson 25 — the canonical
  Adam hyper-parameters from Kingma & Ba (2015), repeated unchanged so that
  reproductions of published results converge identically.

Always profile before tuning: Nsight Compute will tell you whether you are
register-, shared-memory-, or warp-bound for any given kernel.
