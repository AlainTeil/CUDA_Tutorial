# Lesson 17 — End-to-End Training Loop

## Objective

Wire together all previous building blocks (dense layer, ReLU, loss) into
a complete forward → loss → backward → SGD update training cycle.

## Prerequisites

- Lessons 12 (Dense), 13 (Activations), 16 (Loss).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Training loop | Forward → compute loss → backward → update parameters |
| Online SGD | One sample at a time; noisy but simple |
| He initialisation | W ~ N(0, √(2/fan_in)); suited for ReLU networks |
| Backpropagation | Chain rule applied in reverse through the network graph |

## Files

| File | Purpose |
|------|---------|
| `training.cu` | MLP struct, synthetic 3-class data, full training loop |
| `training_test.cu` | Loss decrease, accuracy, softmax sum-to-one, CE non-negativity, SGD weight mutation tests |

## Build & Run

```bash
cmake --build build --target 17_training
./build/lessons/17_training/17_training
```

## Run Tests

```bash
ctest --test-dir build -R 17_training
```

## What You'll Learn

1. How forward, loss, backward, and update compose into a training cycle.
2. Why He initialisation is appropriate for ReLU networks.
3. How synthetic linearly-separable data validates the learning pipeline.
