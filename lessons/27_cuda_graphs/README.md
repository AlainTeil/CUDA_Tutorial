# Lesson 27 — CUDA Graphs

## Objective

Capture GPU operation sequences into replayable graphs to amortise
per-launch CPU overhead — critical for inference pipelines with many small
kernels.

## Prerequisites

- Lesson 07 (Streams).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Stream capture | `cudaStreamBeginCapture` / `EndCapture` records operations into a graph |
| Explicit construction | Build graph node-by-node with `cudaGraphAddKernelNode` |
| Graph update (whole-graph) | Re-bind every changed parameter via `cudaGraphExecUpdate` |
| Graph update (single node) | Mutate one node's params via `cudaGraphExecKernelNodeSetParams` |
| Launch overhead | ~5–10 µs per kernel; amortised to one launch with graphs |

## Parts

- **Part 1** — Stream capture.
- **Part 2** — Explicit graph construction.
- **Part 3** — Graph update. Two complementary APIs are demonstrated:
  `cudaGraphExecUpdate` for re-applying a freshly captured graph, and
  `cudaGraphExecKernelNodeSetParams` for surgically changing one node's
  parameters (e.g. learning rate or step counter) between launches.

## Files

| File | Purpose |
|------|---------|
| `cuda_graphs.cu` | All three graph construction/update approaches |
| `cuda_graphs_test.cu` | Correctness + timing comparison vs non-graph execution |

## Build & Run

```bash
cmake --build build --target 27_cuda_graphs
./build/lessons/27_cuda_graphs/27_cuda_graphs
```

## Run Tests

```bash
ctest --test-dir build -R 27_cuda_graphs
```

## What You'll Learn

1. How stream capture converts a sequence of operations into a graph.
2. How to build graphs explicitly for full control over dependencies.
3. How to update graph parameters without the cost of rebuilding — both
   the whole-graph (`cudaGraphExecUpdate`) and per-node
   (`cudaGraphExecKernelNodeSetParams`) routes.
