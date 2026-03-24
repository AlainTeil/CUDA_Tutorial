# Lesson 32 — Transformer Encoder Block (Capstone)

## Objective

Combine all prior lessons into a complete Transformer encoder block:
embedding → multi-head attention → FFN → layer normalisation → Adam
training — the capstone of the tutorial.

## Prerequisites

- All prior lessons, especially 28 (Embeddings), 29 (Residual + LayerNorm),
  31 (Self-Attention), 25 (Adam).

## Architecture

```
Input IDs → Embedding + PE
         → Multi-Head Self-Attention
         → Residual + LayerNorm
         → FFN (GELU activation)
         → Residual + LayerNorm
         → CLS Pooling → Classification Logits
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| GELU activation | 0.5 · x · (1 + tanh(√(2/π)(x + 0.044715x³))) |
| CLS pooling | Extract the first token's representation for classification |
| Full backward | Gradient back-propagation through attention, FFN, and LayerNorm |
| Adam optimiser | Bias-corrected adaptive learning rates for all parameters |

## Parts

- **Part 1** — Forward kernels (GELU, CLS pool, cross-entropy).
- **Part 2** — Backward kernels (softmax Jacobian, GELU backward).
- **Part 3** — TransformerEncoder struct (forward + backward + Adam).
- **Part 4** — Training loop on synthetic data.

## Files

| File | Purpose |
|------|---------|
| `transformer.cu` | Complete Transformer encoder with training |
| `transformer_test.cu` | GELU, CLS pool, gradient, and forward sanity tests |

## Build & Run

```bash
cmake --build build --target 32_transformer
./build/lessons/32_transformer/32_transformer
```

## Run Tests

```bash
ctest --test-dir build -R 32_transformer
```

## What You'll Learn

1. How all the tutorial's building blocks compose into a real architecture.
2. The complete gradient flow through a Transformer encoder.
3. How GELU compares to ReLU as an activation function.
4. How CLS token pooling produces a fixed-size representation from sequences.
