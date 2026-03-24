# Lesson 15 — Max & Average Pooling

## Objective

Implement max and average pooling with forward and backward passes,
introducing the concept of gradient routing.

## Prerequisites

- Lesson 14 (Conv2D), Lesson 12 (Backward passes).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Max pooling | Selects the maximum in each window; backward routes gradient to argmax |
| Average pooling | Computes the mean; backward distributes gradient equally |
| Gradient routing | Upstream gradients must be scattered to the correct input positions |
| `atomicAdd` | Required when pooling windows overlap (stride < pool size) |

## Files

| File | Purpose |
|------|---------|
| `pooling.cu` | Forward/backward for max and average pooling |
| `pooling_test.cu` | CPU reference, gradient conservation, finite differences |

## Build & Run

```bash
cmake --build build --target 15_pooling
./build/lessons/15_pooling/15_pooling
```

## Run Tests

```bash
ctest --test-dir build -R 15_pooling
```

## What You'll Learn

1. How max pooling achieves translation invariance via down-sampling.
2. Why storing argmax indices is essential for the backward pass.
3. How `atomicAdd` handles overlapping gradient scatter.
