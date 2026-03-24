# Lesson 09 — Prefix Sum (Exclusive Scan)

## Objective

Implement the Blelloch work-efficient exclusive prefix scan — the second
fundamental parallel primitive after reduction.

## Prerequisites

- Lesson 04 (Shared Memory), Lesson 08 (Reduction).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Exclusive scan | `out[i] = sum(in[0..i-1])`; identity element is 0 for addition |
| Up-sweep (reduce) | Computes partial sums up the tree — O(n) work, O(log n) steps |
| Down-sweep | Distributes partial sums down the tree |
| Multi-block extension | Scan block totals recursively, then add to each block's output |

## Files

| File | Purpose |
|------|---------|
| `scan.cu` | Blelloch scan kernel + recursive multi-block driver |
| `scan_test.cu` | CPU reference comparison, all-ones and all-zeros edge cases |

## Build & Run

```bash
cmake --build build --target 09_scan
./build/lessons/09_scan/09_scan
```

## Run Tests

```bash
ctest --test-dir build -R 09_scan
```

## What You'll Learn

1. How the Blelloch algorithm achieves O(n) work-efficient scanning.
2. How to synchronise between up-sweep and down-sweep phases.
3. How to extend a single-block scan to arbitrarily large arrays.
