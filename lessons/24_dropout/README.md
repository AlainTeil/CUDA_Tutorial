# Lesson 24 — Dropout Regularization

## Objective

Implement inverted dropout — randomly zeroing activations during training
to prevent co-adaptation and improve generalisation.

## Prerequisites

- Lesson 17 (Training Loop).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Inverted dropout | Scale remaining activations by 1/(1−p) at train time |
| Hash-based PRNG | Deterministic per-element dropout via hash(seed, index) |
| Mask reuse | Same mask applied in forward and backward for consistency |
| Inference mode | Identity pass-through (no dropout) |

## Files

| File | Purpose |
|------|---------|
| `dropout.cu` | Forward (train/inference), backward, hash-based mask |
| `dropout_test.cu` | Drop rate, scaling, mask consistency, determinism tests |

## Build & Run

```bash
cmake --build build --target 24_dropout
./build/lessons/24_dropout/24_dropout
```

## Run Tests

```bash
ctest --test-dir build -R 24_dropout
```

## What You'll Learn

1. Why inverted dropout preserves the expected activation value.
2. How hash-based PRNG avoids cuRAND dependency while being deterministic.
3. Why the mask must be consistent between forward and backward passes.
