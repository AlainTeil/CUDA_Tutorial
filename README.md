# CUDA Tutorial — GPU Programming for Deep Learning

A comprehensive, self-contained CUDA programming tutorial that progressively
builds from device queries all the way through training and deploying a neural
network entirely on the GPU.  Every lesson includes a pedagogical source file
and a companion unit-test file (powered by Google Test).

## Prerequisites

| Tool | Version |
|------|---------|
| CUDA Toolkit | 13.1+ |
| CMake | 3.20+ |
| GCC | 13+ (Ubuntu 24.04 default) |
| Clang | 18+ (optional, for host compiler) |
| Doxygen | 1.9+ (optional, for docs) |
| clang-format | 18+ (optional, for formatting) |
| clang-tidy | 18+ (optional, for static analysis) |
| Python | 3.8+ (optional, for clang-tidy sanitiser script) |
| Graphviz | any (optional, for Doxygen call graphs) |
| NVIDIA GPU | Compute capability ≥ 7.5 (Turing+) |

## Quick Start

```bash
# Configure (tests ON by default; docs OFF with presets, ON otherwise)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all lessons + tests
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build --output-on-failure -j4

# Run tests for a single lesson
ctest --test-dir build -R 01_device_query --output-on-failure

# Format all source files (requires clang-format)
cmake --build build --target format

# Static analysis (requires clang-tidy + Python 3 + compile_commands.json)
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  # once
cmake --build build --target tidy

# Generate API documentation (requires Doxygen)
cmake --build build --target docs
# → open docs/html/index.html
```

### Using CMake Presets

The project ships a `CMakePresets.json` with ready-made configurations:

```bash
cmake --preset release        # Release build (GCC)
cmake --build --preset release -j$(nproc)
ctest --preset release

cmake --preset debug          # Debug build (GCC)
cmake --preset relwithdebinfo # RelWithDebInfo (optimized + debug symbols)
cmake --preset clang          # Release build (Clang host compiler)
cmake --preset cudnn          # Release + cuDNN (Lesson 19)

# Build & test any preset with the same pattern:
cmake --build --preset <name> -j$(nproc)
ctest --preset <name>
```

Presets automatically enable `CMAKE_EXPORT_COMPILE_COMMANDS` for
clang-tidy, and place build artifacts under `build/<preset-name>/`.

### Building with Clang as host compiler

```bash
# Via preset (recommended):
cmake --preset clang
cmake --build --preset clang -j$(nproc)
ctest --preset clang

# Or manually:
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
| 09 | Prefix Sum (Scan) | Blelloch work-efficient exclusive scan |
| 10 | Matrix Transpose | Bank-conflict-free shared-memory tiling |
| 11 | Tiled Matrix Multiply | Shared-memory tiles, tiled dot-product |

### Phase 3 — Deep Learning Primitives
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 12 | Dense Layer | Forward & backward pass, `Y = XW + b` |
| 13 | Activation Functions | ReLU, Sigmoid, Tanh, Softmax (forward + backward) |
| 14 | 2-D Convolution | Direct convolution, im2col + GEMM |
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

### Phase 5 — Advanced CUDA & Deep Learning
| # | Lesson | Key Concepts |
|---|--------|-------------|
| 23 | Batch Normalization | Mean/variance kernels, forward/backward, running stats |
| 24 | Dropout | Hash-based PRNG (Philox-style), inverted scaling, mask generation |
| 25 | Optimizers | SGD + momentum, Adam, AdamW, cosine & warmup LR schedules |
| 26 | Cooperative Groups | `tiled_partition`, `cg::reduce`, `coalesced_threads`, grid-level sync |
| 27 | CUDA Graphs | Stream capture, explicit graph construction, graph update |
| 28 | Embeddings | Token embedding gather/scatter, sinusoidal PE, cuBLAS projection |
| 29 | Residual + LayerNorm | Residual connections, Welford's online algorithm, fused kernel |
| 30 | GRU | Gated Recurrent Unit, cuBLAS GEMM, BPTT backward loop |
| 31 | Self-Attention | Multi-head attention, split/merge heads, batched GEMM, softmax |
| 32 | Transformer Encoder | Capstone: full encoder block, backward pass, Adam optimizer, end-to-end training |

## Project Structure

```
CUDA_Tutorial/
├── CMakeLists.txt          # Root build configuration
├── cmake/
│   ├── CompilerWarnings.cmake
│   ├── run_clang_tidy.py          # Sanitises nvcc flags for clang-tidy
│   └── clang_cuda_compat/
│       └── texture_fetch_functions.h  # Stub for Clang 18 + CUDA 13
├── .clang-format           # Google style
├── .clang-tidy             # Static analysis configuration
├── .gitignore
├── CMakePresets.json        # Named build presets
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
    ├── ...
    ├── 23_batch_normalization/
    ├── 24_dropout/
    ├── 25_optimizers/
    ├── 26_cooperative_groups/
    ├── 27_cuda_graphs/
    ├── 28_embeddings/
    ├── 29_residual_layernorm/
    ├── 30_gru/
    ├── 31_self_attention/
    └── 32_transformer/
```

## Troubleshooting

<details>
<summary><strong>CMake cannot find CUDA</strong></summary>

Ensure `nvcc` is on your `PATH`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```
If you installed the toolkit via the runfile, make sure
`/usr/local/cuda` is a symlink to the correct version.
</details>

<details>
<summary><strong>"no kernel image is available for execution on the device"</strong></summary>

Your GPU's compute capability is not in `CMAKE_CUDA_ARCHITECTURES`.
Reconfigure with the correct SM version:
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release
```
Run `nvidia-smi` or Lesson 01 to discover your GPU's SM version.
</details>

<details>
<summary><strong>Cooperative groups tests fail (Lesson 26)</strong></summary>

`cudaLaunchCooperativeKernel` requires that the grid size does not
exceed the device's occupancy limit.  If you changed the block size or
data size, reduce the grid until
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` is satisfied.
</details>

<details>
<summary><strong>cuDNN not found (Lesson 19)</strong></summary>

Ensure the cuDNN headers and libraries are under
`CUDAToolkit_LIBRARY_DIR` (typically `/usr/local/cuda/lib64`):
```bash
ls /usr/local/cuda/include/cudnn.h
ls /usr/local/cuda/lib64/libcudnn.so
```
If cuDNN is installed elsewhere, pass `-DCUDNN_PATH=/path/to/cudnn`.
</details>

<details>
<summary><strong>Link error: "undefined reference to cublasCreate"</strong></summary>

This usually means `find_package(CUDAToolkit)` failed silently.
Verify CUDA Toolkit installation and that CMake ≥ 3.20 is in use.
</details>

<details>
<summary><strong>clang-format / clang-tidy targets missing</strong></summary>

The `format` and `tidy` targets are only generated when the
corresponding tool is found at configure time.  The `tidy` target also
requires Python 3.  Install them with:
```bash
sudo apt install clang-format clang-tidy python3   # Ubuntu/Debian
```
Then re-run `cmake -B build …`.

The `tidy` target uses `cmake/run_clang_tidy.py` to sanitise the
nvcc-generated `compile_commands.json` so that Clang's frontend can
parse it.  This is fully automatic — just run `cmake --build build --target tidy`.
</details>

## License

This tutorial is provided for educational purposes.
