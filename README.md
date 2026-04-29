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
# Configure (tests ON by default; docs target OFF unless opted in)
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

# Generate API documentation (requires Doxygen).
# The 'docs' target is only created when CUDA_TUTORIAL_BUILD_DOCS=ON, so
# (re)configure with the option before building it:
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_TUTORIAL_BUILD_DOCS=ON
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
cmake --preset ubsan          # RelWithDebInfo + host UBSan (see Sanitizers)

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

### Sanitizers

Two complementary tools catch runtime bugs cheaply:

| Scope | Tool | How to use |
|-------|------|------------|
| **Host code** (allocations, signed-overflow, OOB shifts, null derefs, …) | UBSan | `cmake --preset ubsan` → build & test as usual |
| **Device code** (illegal accesses, races, uninitialised reads, sync bugs) | NVIDIA `compute-sanitizer` | wrap any test binary at run-time |

```bash
# Host-side UndefinedBehaviorSanitizer (any test failing UB aborts immediately):
cmake --preset ubsan
cmake --build --preset ubsan -j$(nproc)
ctest --preset ubsan

# Device-side checks via compute-sanitizer (build with relwithdebinfo for line info):
cmake --preset relwithdebinfo
cmake --build --preset relwithdebinfo -j$(nproc)
compute-sanitizer --tool memcheck   ./build/relwithdebinfo/lessons/08_reduction/08_reduction_test
compute-sanitizer --tool racecheck  ./build/relwithdebinfo/lessons/10_transpose/10_transpose_test
compute-sanitizer --tool initcheck  ./build/relwithdebinfo/lessons/14_conv2d/14_conv2d_test
compute-sanitizer --tool synccheck  ./build/relwithdebinfo/lessons/26_cooperative_groups/26_cooperative_groups_test
```

ASan is not wired up because it interacts unreliably with the CUDA runtime
initialisation path on many setups; UBSan + `compute-sanitizer` cover the
practical bug surface.

## Lesson Index

> **Difficulty legend** — ★ beginner · ★★ easy · ★★★ moderate · ★★★★ advanced · ★★★★★ expert.
> **Time** is a rough estimate to read, build, and step through the test cases on a modern workstation GPU.

### Phase 1 — CUDA Fundamentals
| # | Lesson | Difficulty | Time | Key Concepts | See also |
|---|--------|:-:|:-:|-------------|---------|
| 01 | Device Query | ★ | 15 min | `cudaGetDeviceProperties`, inspecting GPU specs | — |
| 02 | Hello Kernel | ★ | 20 min | `__global__`, `<<<>>>`, `cudaMalloc`, `cudaMemcpy` | 03, 04 |
| 03 | Error Handling | ★ | 15 min | `CUDA_CHECK` macro, `cudaGetLastError` | 02 |
| 04 | Memory Model | ★★ | 30 min | Global, `__constant__`, `__shared__` memory | 05, 08 |
| 05 | Thread Hierarchy | ★★ | 30 min | 1-D/2-D/3-D grids & blocks, `blockIdx`, `threadIdx` | 04, 11 |
| 06 | Memory Transfers | ★★ | 30 min | Pinned memory, unified memory, bandwidth measurement | 07, 22 |
| 07 | Streams & Events | ★★ | 30 min | Async execution, overlapping compute & transfer | 06, 27 |

### Phase 2 — Parallel Patterns
| # | Lesson | Difficulty | Time | Key Concepts | See also |
|---|--------|:-:|:-:|-------------|---------|
| 08 | Parallel Reduction | ★★★ | 45 min | Shared-memory tree reduction, `__shfl_down_sync` | 04, 09, 26 |
| 09 | Prefix Sum (Scan) | ★★★ | 45 min | Blelloch work-efficient exclusive scan | 08 |
| 10 | Matrix Transpose | ★★★ | 30 min | Bank-conflict-free shared-memory tiling | 04, 11 |
| 11 | Tiled Matrix Multiply | ★★★ | 60 min | Shared-memory tiles, tiled dot-product | 10, 14, 19, 20 |

### Phase 3 — Deep Learning Primitives
| # | Lesson | Difficulty | Time | Key Concepts | See also |
|---|--------|:-:|:-:|-------------|---------|
| 12 | Dense Layer | ★★★ | 45 min | Forward & backward pass, `Y = XW + b` | 11, 16, 17 |
| 13 | Activation Functions | ★★ | 30 min | ReLU, Sigmoid, Tanh, Softmax (forward + backward) | 12, 16 |
| 14 | 2-D Convolution | ★★★★ | 75 min | Direct convolution, im2col + GEMM, backward (dkernel, dx) | 11, 15, 19 |
| 15 | Pooling Layers | ★★★ | 30 min | Max-pool & average-pool forward/backward, index tracking | 14 |
| 16 | Loss Functions | ★★★ | 30 min | MSE, log-softmax, cross-entropy (forward + backward) | 13, 17 |

### Phase 4 — Training & Inference
| # | Lesson | Difficulty | Time | Key Concepts | See also |
|---|--------|:-:|:-:|-------------|---------|
| 17 | Training Loop | ★★★ | 60 min | 2-layer MLP, SGD optimizer, synthetic 3-class data | 12, 16, 20, 25 |
| 18 | Inference Pipeline | ★★ | 30 min | Binary weight serialization, save/load, batched inference | 17 |
| 19 | cuBLAS & cuDNN | ★★★ | 45 min | `cublasSgemm`, cuDNN convolution (optional build) | 11, 14, 20 |
| 20 | Mini-Batch Training | ★★★★ | 75 min | cuBLAS GEMM batched forward/backward, gradient averaging | 17, 19 |
| 21 | Mixed Precision | ★★★★ | 60 min | FP16 `__half`, TF32 Tensor Cores, `cublasGemmEx`, loss scaling | 19, 20 |
| 22 | Performance Measurement | ★★★ | 45 min | `cudaEvent_t` benchmarking, bandwidth/GFLOPS, roofline model | 06, 07 |

### Phase 5 — Advanced CUDA & Deep Learning
| # | Lesson | Difficulty | Time | Key Concepts | See also |
|---|--------|:-:|:-:|-------------|---------|
| 23 | Batch Normalization | ★★★★ | 60 min | Mean/variance kernels, forward/backward, running stats | 13, 29 |
| 24 | Dropout | ★★★ | 45 min | Hash-based PRNG (Philox-style), inverted scaling, mask generation | 13 |
| 25 | Optimizers | ★★★ | 60 min | SGD + momentum, Adam, AdamW, cosine & warmup LR schedules | 17, 20 |
| 26 | Cooperative Groups | ★★★★ | 60 min | `tiled_partition`, `cg::reduce`, `coalesced_threads`, grid-level sync | 08 |
| 27 | CUDA Graphs | ★★★★ | 45 min | Stream capture, explicit graph construction, graph update | 07 |
| 28 | Embeddings | ★★★ | 45 min | Token embedding gather/scatter, sinusoidal PE, cuBLAS projection | 31, 32 |
| 29 | Residual + LayerNorm | ★★★★ | 60 min | Residual connections, Welford's online algorithm, fused kernel | 23, 32 |
| 30 | GRU | ★★★★★ | 90 min | Gated Recurrent Unit, cuBLAS GEMM, BPTT backward loop | 19, 25 |
| 31 | Self-Attention | ★★★★★ | 90 min | Multi-head attention forward + backward, split/merge heads, batched GEMM, softmax | 13, 19, 32 |
| 32 | Transformer Encoder | ★★★★★ | 120 min | Capstone: full encoder block, backward pass, Adam optimizer, end-to-end training | 25, 28, 29, 31 |

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
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: format check, CPU build, clang-tidy, GPU test
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

## Continuous Integration

The project ships a [GitHub Actions workflow](.github/workflows/ci.yml) with
five jobs:

| Job | Runner | Purpose |
|-----|--------|---------|
| `format-check` | `ubuntu-latest` | `clang-format --dry-run --Werror` over every source file |
| `build-cpu` | `ubuntu-24.04` | Configures + compiles every lesson with CUDA 12.6 (multi-arch); uploads `compile_commands.json` as an artifact. Tests are built but not executed (no GPU on hosted runners). |
| `build-cudnn` | `ubuntu-24.04` | Same as `build-cpu` but with `CUDA_TUTORIAL_USE_CUDNN=ON` (installs `libcudnn9-dev-cuda-12`). Guards against accidental regressions in the optional Lesson 19 cuDNN code path. |
| `clang-tidy` | `ubuntu-24.04` | Runs `cmake/run_clang_tidy.py` on a representative subset of lessons after `build-cpu` succeeds. |
| `build-and-test` | self-hosted `[linux, gpu]` | Configures with `--preset release`, builds, and runs the full `ctest` suite on a real GPU. Skipped automatically if no self-hosted runner is registered. |

To enable the GPU job, register a runner with the labels `self-hosted`,
`linux`, `gpu`, with the CUDA Toolkit and an NVIDIA driver installed.

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
