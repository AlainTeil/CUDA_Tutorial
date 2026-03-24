# Lesson 06 — Memory Transfers

## Objective

Compare three host memory allocation strategies — pageable, pinned, and
unified — and their impact on PCIe transfer performance.

## Prerequisites

- Lesson 04 (Memory Model).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Pageable memory | `std::vector` / `malloc`; may be paged out; driver uses a staging buffer |
| Pinned memory | `cudaMallocHost`; DMA-capable; ~2× faster transfers |
| Unified memory | `cudaMallocManaged`; automatic page migration; variable performance |
| CUDA events | Sub-millisecond GPU-side timing via `cudaEventElapsedTime` |

## Files

| File | Purpose |
|------|---------|
| `memory_transfers.cu` | Runs all three paths with event-based timing |
| `memory_transfers_test.cu` | Verifies identical results across all three strategies |

## Build & Run

```bash
cmake --build build --target 06_memory_transfers
./build/lessons/06_memory_transfers/06_memory_transfers
```

## Run Tests

```bash
ctest --test-dir build -R 06_memory_transfers
```

## What You'll Learn

1. Why pinned memory enables faster DMA transfers.
2. Trade-offs of unified memory (simplicity vs performance predictability).
3. How to use CUDA events for accurate GPU timing.
