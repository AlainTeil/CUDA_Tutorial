# Lesson 30 — Gated Recurrent Unit (GRU)

## Objective

Implement the GRU recurrent cell (Cho et al., 2014) with forward
unrolling over a sequence and full BPTT backward pass computing
gradients for weights (dW, dU), biases, and inputs.

## Prerequisites

- Lesson 19 (cuBLAS), Lesson 13 (Sigmoid, Tanh).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Update gate (z) | Controls how much of the previous hidden state to keep |
| Reset gate (r) | Controls how much of the previous state to forget |
| Candidate (ñ) | New candidate hidden state |
| Concatenated GEMM | W_z, W_r, W_n packed into (3H × I) for a single GEMM |
| BPTT | Back-Propagation Through Time — unroll backward over timesteps |

## Parts

- **Part 1** — Forward: single-step GRU cell.
- **Part 2** — Forward: unrolled over T time-steps.
- **Part 3** — Backward: full BPTT through T steps (dW, dU, dbias, dx, dh).

## Files

| File | Purpose |
|------|---------|
| `gru.cu` | GRU cell, sequence unrolling, full BPTT |
| `gru_test.cu` | Gate correctness, hidden-state evolution, bias-gradient sanity, BPTT weight (dW, dU, dbias) and input (dx) gradients verified against central finite differences |

## Build & Run

```bash
cmake --build build --target 30_gru
./build/lessons/30_gru/30_gru
```

## Run Tests

```bash
ctest --test-dir build -R 30_gru
```

## What You'll Learn

1. How GRU's two gates (update, reset) control information flow.
2. How concatenated GEMMs reduce the number of cuBLAS calls.
3. The basic structure of BPTT and gate-level gradient computation.
4. How to validate a hand-written BPTT against central finite differences
   for both weights and inputs.
