# Lesson 07 — Streams & Events

## Objective

Learn how CUDA streams enable overlapping of memory transfers and kernel
execution for latency hiding.

## Prerequisites

- Lesson 06 (Memory Transfers — pinned memory).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Default stream | Stream 0; all operations are serialized |
| Named streams | `cudaStreamCreate`; independent streams can run concurrently |
| `cudaMemcpyAsync` | Non-blocking copy that requires pinned host memory |
| Stream parameter | `<<<grid, block, smem, stream>>>` routes a kernel to a stream |

## Files

| File | Purpose |
|------|---------|
| `streams_events.cu` | Serial vs concurrent execution with timing |
| `streams_events_test.cu` | Verifies both paths produce identical results |

## Build & Run

```bash
cmake --build build --target 07_streams_events
./build/lessons/07_streams_events/07_streams_events
```

## Run Tests

```bash
ctest --test-dir build -R 07_streams_events
```

## What You'll Learn

1. How to create and destroy CUDA streams.
2. How to overlap H→D copy, kernel, and D→H copy across streams.
3. Requirements for overlap (pinned memory, multiple copy engines).
4. How events measure per-stream elapsed time.
