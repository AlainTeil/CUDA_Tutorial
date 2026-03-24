# Lesson 22 — Performance Measurement & Benchmarking

## Objective

Systematically benchmark GPU kernels using CUDA events, compute effective
bandwidth and GFLOPS, and classify kernels via the roofline model.

## Prerequisites

- Lessons 01–11 (basic kernels to benchmark).

## Key Concepts

| Concept | Description |
|---------|-------------|
| `GpuTimer` | RAII wrapper around `cudaEvent_t` for sub-microsecond timing |
| Effective bandwidth | GB/s = total bytes / median time |
| Arithmetic throughput | GFLOPS = total FLOPs / median time |
| Roofline model | Ridge point = peak GFLOPS / peak BW; classifies memory- vs compute-bound |
| Warm-up | First launches are slower; always discard initial iterations |

## Files

| File | Purpose |
|------|---------|
| `performance_measurement.cu` | GpuTimer, BenchmarkResult, roofline helpers, example kernels |
| `performance_measurement_test.cu` | Timer, stats, bandwidth, GFLOPS, roofline, kernel correctness, edge-case, and scaling tests |

## Build & Run

```bash
cmake --build build --target 22_performance_measurement
./build/lessons/22_performance_measurement/22_performance_measurement
```

## Run Tests

```bash
ctest --test-dir build -R 22_performance_measurement
```

## What You'll Learn

1. How to measure kernel execution time accurately with CUDA events.
2. How to compute effective memory bandwidth and arithmetic throughput.
3. How the roofline model determines the bottleneck for any kernel.
4. Why warm-up iterations are essential for reliable benchmarks.

## Note — CUDA 13+ Compatibility

The `cudaDevAttrMemoryClockRate` and `cudaDevAttrClockRate` attributes were
deprecated in CUDA 13. The code queries them gracefully and falls back to a
conservative 1 GHz default if the driver returns an error, printing a
warning to stderr.
