# Lesson 18 — Inference Pipeline

## Objective

Complete the ML lifecycle: train → save weights → load weights → infer.
Demonstrates binary weight serialisation and batched prediction.

## Prerequisites

- Lesson 17 (Training Loop).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Weight serialisation | Binary format with dimension header prevents mismatches |
| Save / Load | Separate processes for training and deployment |
| Batched inference | Process multiple inputs (naive loop; optimised in Lesson 20) |
| `argmax` | No softmax needed for prediction — argmax is monotone |

## Files

| File | Purpose |
|------|---------|
| `inference.cu` | InferenceMLP with save/load/predict_batch |
| `inference_test.cu` | Save/load round-trip, accuracy, file size, logit finiteness, determinism, loss non-negativity tests |

## Build & Run

```bash
cmake --build build --target 18_inference
./build/lessons/18_inference/18_inference
```

## Run Tests

```bash
ctest --test-dir build -R 18_inference
```

## What You'll Learn

1. How to serialise and deserialise model weights in binary format.
2. Why dimension checking prevents silent model-mismatch bugs.
3. That one-sample-at-a-time inference is inefficient (fixed in Lesson 20).
