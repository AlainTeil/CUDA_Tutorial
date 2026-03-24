# Lesson 01 — Device Query

## Objective

Learn how to query and display CUDA device properties before writing any
GPU code. Understanding your hardware's capabilities is the foundation for
writing efficient kernels.

## Prerequisites

- A working CUDA Toolkit installation.
- Basic C++ knowledge.

## Key Concepts

| Concept | Description |
|---------|-------------|
| `cudaGetDeviceCount` | Discover how many CUDA-capable GPUs are available |
| `cudaGetDeviceProperties` | Retrieve the full `cudaDeviceProp` structure |
| Compute capability | Determines which features the GPU supports |
| Hardware limits | Max threads per block, warp size, shared memory size |

## Files

| File | Purpose |
|------|---------|
| `device_query.cu` | Queries and prints all device properties |
| `device_query_test.cu` | Validates device properties meet minimum requirements |

## Build & Run

```bash
cmake --build build --target 01_device_query
./build/lessons/01_device_query/01_device_query
```

## Run Tests

```bash
ctest --test-dir build -R 01_device_query
```

## What You'll Learn

1. How to enumerate CUDA devices on the system.
2. How to read hardware limits (max threads, shared memory, warp size).
3. Why checking compute capability matters for feature support.
4. How to select a device for subsequent kernel launches.
