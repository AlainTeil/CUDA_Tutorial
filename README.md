# CUDA Tutorial — GPU Programming for Deep Learning

A comprehensive, self-contained CUDA programming tutorial that progressively
builds from device queries all the way through training and deploying a neural
network entirely on the GPU.  Every lesson includes a pedagogical source file
and a companion unit-test file (188 tests in total, powered by Google Test).

## Prerequisites

| Tool | Version |
|------|---------|
| CUDA Toolkit | 13.1+ |
| CMake | 3.20+ |
| GCC | 13+ (Ubuntu 24.04 default) |
| Clang | 18+ (optional, for host compiler) |
| Doxygen | 1.9+ (optional, for docs) |
| clang-format | 18+ (optional, for formatting) |
| Graphviz | any (optional, for Doxygen call graphs) |
| NVIDIA GPU | Compute capability ≥ 7.5 (Turing+) |

## Quick Start

```bash
# Configure (tests and docs targets are ON by default)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all lessons + tests
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build --output-on-failure -j4

# Format all source files (requires clang-format)
cmake --build build --target format

# Generate API documentation (requires Doxygen)
cmake --build build --target docs
# → open docs/html/index.html
```

### Building with Clang as host compiler

```bash
cmake -B build-clang \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-clang -j$(nproc)
ctest --test-dir build-clang --output-on-failure -j4
```

CMake automatically tells NVCC to use the configured CXX compiler as the
CUDA host compiler, so there is no need to set `CMAKE_CUDA_HOST_COMPILER`
separately.

### Enabling cuBLAS / cuDNN (Lesson 19)

```bash
cmake -B build -DCUDA_TUTORIAL_USE_CUDNN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Targeting additional architectures (e.g. Blackwell)

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100" \
  -DCMAKE_BUILD_TYPE=Release
```

## Lesson Index

### Phase 1 — CUDA Fundamentals
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 01 | Device Query | `cudaGetDeviceProperties`, inspecting GPU specs |
| 02 | Hello Kernel | `__global__`, `<<<>>>`, `cudaMalloc`, `cudaMemcpy` |
| 03 | Error Handling | `CUDA_CHECK` macro, `cudaGetLastError` |
| 04 | Memory Model | Global, `__constant__`, `__shared__` memory |
| 05 | Thread Hierarchy | 1-D/2-D/3-D grids & blocks, `blockIdx`, `threadIdx` |
| 06 | Memory Transfers | Pinned memory, unified memory, bandwidth measurement |
| 07 | Streams & Events | Async execution, overlapping compute & transfer |

### Phase 2 — Parallel Patterns
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 08 | Parallel Reduction | Shared-memory tree reduction, `__shfl_down_sync` |
| 09 | Prefix Sum (Scan) | Blelloch inclusive/exclusive scan |
| 10 | Matrix Transpose | Bank-conflict-free shared-memory tiling |
| 11 | Tiled Matrix Multiply | Shared-memory tiles, register blocking |

### Phase 3 — Deep Learning Primitives
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 12 | Dense Layer | Forward & backward pass, `Y = XW + b` |
| 13 | Activation Functions | ReLU, Sigmoid, Tanh, Softmax (forward + backward) |
| 14 | 2-D Convolution | Direct convolution kernel |
| 15 | Pooling Layers | Max-pool & average-pool forward/backward, index tracking |
| 16 | Loss Functions | MSE, log-softmax, cross-entropy (forward + backward) |

### Phase 4 — Training & Inference
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 17 | Training Loop | 2-layer MLP, SGD optimizer, synthetic 3-class data |
| 18 | Inference Pipeline | Binary weight serialization, save/load, batched inference |
| 19 | cuBLAS & cuDNN | `cublasSgemm`, cuDNN convolution (optional build) |
| 20 | Mini-Batch Training | cuBLAS GEMM batched forward/backward, gradient averaging |
| 21 | Mixed Precision | FP16 `__half`, TF32 Tensor Cores, `cublasGemmEx`, loss scaling |
| 22 | Performance Measurement | `cudaEvent_t` benchmarking, bandwidth/GFLOPS, roofline model |

## Project Structure

```
CUDA_Tutorial/
├── CMakeLists.txt          # Root build configuration
├── cmake/
│   └── CompilerWarnings.cmake
├── .clang-format           # Google style
├── .gitignore
├── Doxyfile                # Doxygen 1.9.8 configuration
├── README.md
└── lessons/
    ├── CMakeLists.txt      # Adds each lesson subdirectory
    ├── 01_device_query/
    │   ├── CMakeLists.txt
    │   ├── device_query.cu
    │   └── device_query_test.cu
    ├── 02_hello_kernel/
    │   └── ...
    └── ...
```

## License

This tutorial is provided for educational purposes.
