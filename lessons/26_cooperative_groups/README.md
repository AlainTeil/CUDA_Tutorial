# Lesson 26 — Cooperative Groups

## Objective

Modernise synchronisation primitives using the Cooperative Groups API —
type-safe thread groups that replace raw `__syncthreads()` and shuffle
masks.

## Prerequisites

- Lesson 08 (Reduction), Lesson 04 (Shared Memory).

## Key Concepts

| Concept | Description |
|---------|-------------|
| `thread_block_tile<32>` | Compile-time warp-sized tile for safe reductions |
| `cg::reduce()` | Built-in templated reduction (one-liner) |
| `coalesced_threads()` | Dynamic groups after divergent branches |
| `grid_group` + `grid.sync()` | Grid-wide barrier for single-pass global reductions |

## Files

| File | Purpose |
|------|---------|
| `cooperative_groups.cu` | Tile, builtin, coalesced, and grid-level reductions |
| `cooperative_groups_test.cu` | CPU reference comparison across multiple sizes |

## Build & Run

```bash
cmake --build build --target 26_cooperative_groups
./build/lessons/26_cooperative_groups/26_cooperative_groups
```

## Run Tests

```bash
ctest --test-dir build -R 26_cooperative_groups
```

## What You'll Learn

1. How Cooperative Groups replace `__syncthreads()` with type-safe APIs.
2. How `grid.sync()` enables single-pass grid-wide reductions.
3. Why cooperative launches require all blocks to be concurrently active.
4. How `coalesced_threads()` works for dynamically formed thread groups.
