# Lesson 16 — Loss Functions

## Objective

Implement MSE (regression) and cross-entropy with log-softmax
(classification) loss functions — the training signal for neural networks.

## Prerequisites

- Lesson 13 (Softmax), Lesson 08 (Reduction).

## Key Concepts

| Concept | Description |
|---------|-------------|
| MSE | (1/N) Σ(pred − target)²; backward is 2(pred − target) / N |
| Log-softmax | log(softmax(x)) computed stably via max-subtraction |
| Cross-entropy | −Σ target · log_softmax; the standard classification loss |
| Fused gradient | softmax(x) − target: the elegant combined CE+softmax gradient |

## Files

| File | Purpose |
|------|---------|
| `loss.cu` | MSE and cross-entropy forward/backward + log-softmax |
| `loss_test.cu` | Known-value tests, property checks, finite-difference gradients |

## Build & Run

```bash
cmake --build build --target 16_loss
./build/lessons/16_loss/16_loss
```

## Run Tests

```bash
ctest --test-dir build -R 16_loss
```

## What You'll Learn

1. Why softmax and cross-entropy are always paired (simple gradient form).
2. The log-sum-exp trick for numerical stability.
3. How shared-memory reductions implement per-row log-softmax on the GPU.
