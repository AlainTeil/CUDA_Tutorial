/**
 * @file performance_measurement.cu
 * @brief Lesson 22 — Performance Measurement & Benchmarking.
 *
 * Writing a fast kernel is only half the job — **measuring** how fast it is
 * (and how close to the hardware limit) is equally important.  This lesson
 * teaches the tools and mental models used to evaluate CUDA kernel performance.
 *
 * ## Topics covered
 *
 * 1. **`cudaEvent_t` benchmarking** — GPU-side sub-millisecond timing with
 *    warm-up iterations, multiple trials, and cold-cache avoidance.
 *
 * 2. **Effective bandwidth** — For memory-bound kernels (the majority),
 *    the key metric is GB/s:
 *      @code
 *      bandwidth_GB_s = bytes_transferred / (elapsed_s * 1e9)
 *      @endcode
 *    Compare this to the GPU's theoretical peak (memoryClockRate × busWidth
 *    × 2 for DDR) to see how efficiently you use the memory subsystem.
 *
 * 3. **Arithmetic throughput (GFLOPS)** — For compute-bound kernels,
 *    measure floating-point operations per second:
 *      @code
 *      gflops = flop_count / (elapsed_s * 1e9)
 *      @endcode
 *
 * 4. **Roofline model** — A visual way to tell whether a kernel is
 *    memory-bound or compute-bound.  The **arithmetic intensity** (AI)
 *    is the ratio flops / bytes.  The "roofline" has two regimes:
 *
 *      - AI < ridge point → **memory-bound** (bandwidth ceiling).
 *      - AI > ridge point → **compute-bound** (FLOP ceiling).
 *
 *    The ridge point = peak_GFLOPS / peak_BW_GB_s.
 *
 * 5. **Practical benchmarking pitfalls** — Why you must warm up the GPU,
 *    run multiple trials, synchronise properly, and avoid measuring
 *    host-side overhead.
 *
 * ## Relation to earlier lessons
 *
 * | Lesson | What it introduced          | This lesson extends it with        |
 * |--------|-----------------------------|-------------------------------------|
 * | 07     | `cudaEvent_t` basics        | Reusable `GpuTimer`, warm-up, stats |
 * | 06     | Bandwidth concept           | Effective-BW formula, % of peak     |
 * | 08     | Reduction kernel            | Benchmark it, compute BW            |
 * | 11     | Tiled matmul                | Benchmark it, compute GFLOPS        |
 *
 * Build: requires only CUDA runtime (no cuBLAS/cuDNN).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

// =============================================================================
// Error-checking macro
// =============================================================================

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// =============================================================================
// GpuTimer — reusable RAII event-based timer
// =============================================================================

/// @brief RAII wrapper around `cudaEvent_t` pairs for GPU-side timing.
///
/// ### Why GPU events, not `std::chrono`?
///
/// `std::chrono` measures **wall-clock time** including driver overhead,
/// kernel launch latency, and host-side work.  `cudaEvent_t` is inserted
/// into the GPU command stream and records timestamps on the GPU itself,
/// giving sub-microsecond precision for the kernel execution only.
///
/// ### Usage pattern
/// @code
///   GpuTimer timer;
///   timer.start();
///   my_kernel<<<grid, blk>>>(...);
///   timer.stop();
///   float ms = timer.elapsed_ms();
/// @endcode
///
/// ### Important: warm-up
///
/// The first kernel launch after program start (or after a long idle
/// period) is slower because the GPU must:
///   - Boost clock frequencies from idle.
///   - Populate instruction caches.
///   - Page in device memory.
///
/// Always run a few **warm-up iterations** before measuring.
struct GpuTimer {
  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  // Non-copyable.
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  /// Record the start timestamp on the default stream.
  void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_event, stream)); }

  /// Record the stop timestamp on the default stream.
  void stop(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(stop_event, stream)); }

  /// Block until the stop event completes, then return elapsed time in ms.
  ///
  /// @note This synchronises the host with the GPU — do NOT call in a
  /// latency-critical path.
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return ms;
  }
};

// =============================================================================
// BenchmarkResult — collects timing statistics
// =============================================================================

/// @brief Stores results from benchmarking a kernel over multiple trials.
///
/// Capturing min, max, and median gives a complete picture:
///   - **Median** — robust central tendency (ignores outliers).
///   - **Min**    — best-case (closest to hardware limit).
///   - **Max**    — worst-case (includes scheduling jitter).
struct BenchmarkResult {
  float min_ms{};
  float max_ms{};
  float median_ms{};
  float mean_ms{};
  int trials{};

  /// Compute derived metrics.
  /// @param bytes  Total bytes read + written by the kernel.
  /// @return Effective bandwidth in GB/s, using the **median** time.
  float bandwidth_gb_s(size_t bytes) const {
    return static_cast<float>(bytes) / (median_ms * 1e6F);  // ms → s, bytes → GB
  }

  /// @param flops  Total floating-point operations.
  /// @return Arithmetic throughput in GFLOPS, using the **median** time.
  float gflops(size_t flops) const { return static_cast<float>(flops) / (median_ms * 1e6F); }
};

/// Compute `BenchmarkResult` from a vector of per-trial timings.
static BenchmarkResult compute_stats(std::vector<float>& times) {
  std::sort(times.begin(), times.end());
  BenchmarkResult r;
  r.trials = static_cast<int>(times.size());
  r.min_ms = times.front();
  r.max_ms = times.back();
  r.mean_ms = std::accumulate(times.begin(), times.end(), 0.0F) / static_cast<float>(times.size());

  // Median.
  size_t n = times.size();
  if (n % 2 == 0) {
    r.median_ms = (times[n / 2 - 1] + times[n / 2]) / 2.0F;
  } else {
    r.median_ms = times[n / 2];
  }
  return r;
}

// =============================================================================
// Theoretical peak helpers
// =============================================================================

/// @brief Query the GPU's theoretical peak memory bandwidth in GB/s.
///
/// Formula: BW = memoryClockRate (kHz) × (memoryBusWidth / 8) × 2 (DDR) / 1e6.
///
/// This is the absolute upper bound — real kernels can't quite reach it
/// because of ECC overhead, memory controller efficiency, and refresh cycles.
/// Achieving **80–90%** of peak is excellent.
static float theoretical_peak_bw_gb_s(int device = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  // memoryClockRate was removed from cudaDeviceProp in CUDA 13; query via attribute.
  int mem_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
  // mem_clock_khz is in kHz, memoryBusWidth in bits.
  double bw = static_cast<double>(mem_clock_khz) * 1e3            // kHz → Hz
              * (static_cast<double>(prop.memoryBusWidth) / 8.0)  // bits → bytes
              * 2.0                                               // DDR
              / 1e9;                                              // → GB/s
  return static_cast<float>(bw);
}

/// @brief Estimate peak single-precision GFLOPS.
///
/// Rough estimate: SMs × clockRate × 2 (FMA = 2 FLOP) × CUDACoresPerSM / 1e6.
/// The cores-per-SM value depends on architecture — we use a conservative
/// approximation from the compute capability.
static float estimated_peak_gflops(int device = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  // Approximate CUDA cores per SM based on compute capability.
  int cores_per_sm;
  if (prop.major == 7 && prop.minor == 0) {
    cores_per_sm = 64;  // Volta  (sm_70)
  } else if (prop.major == 7 && prop.minor == 5) {
    cores_per_sm = 64;  // Turing (sm_75)
  } else if (prop.major == 8 && prop.minor == 0) {
    cores_per_sm = 64;  // Ampere A100 (sm_80)
  } else if (prop.major == 8 && prop.minor == 6) {
    cores_per_sm = 128;  // Ampere GA10x (sm_86)
  } else if (prop.major == 8 && prop.minor == 9) {
    cores_per_sm = 128;  // Ada Lovelace (sm_89)
  } else if (prop.major == 9 && prop.minor == 0) {
    cores_per_sm = 128;  // Hopper (sm_90)
  } else {
    cores_per_sm = 64;  // Conservative fallback.
  }

  // clockRate was removed from cudaDeviceProp in CUDA 13; query via attribute.
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  double gflops = static_cast<double>(prop.multiProcessorCount) *
                  static_cast<double>(cores_per_sm) * static_cast<double>(clock_khz) *
                  1e3     // kHz → Hz
                  * 2.0   // FMA = 2 FLOP
                  / 1e9;  // → GFLOPS
  return static_cast<float>(gflops);
}

/// @brief Compute the roofline ridge point.
///
/// Ridge point = peak_GFLOPS / peak_BW_GB_s.
/// Below this AI value, the kernel is memory-bound.
/// Above it, compute-bound.
static float roofline_ridge_point(int device = 0) {
  return estimated_peak_gflops(device) / theoretical_peak_bw_gb_s(device);
}

// =============================================================================
// Example kernels to benchmark
// =============================================================================

/// @defgroup benchmark_kernels Benchmark example kernels
/// @{

/// **Vector add** — classic memory-bound kernel.
///
/// Each element requires: 2 loads (8 bytes) + 1 store (4 bytes) = 12 bytes,
/// and 1 FLOP (addition).  Arithmetic intensity = 1/12 ≈ 0.083 FLOP/byte,
/// far below even the lowest ridge point.  → **Memory-bound**.
__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

/// **Saxpy** (y = a·x + y) — also memory-bound.
///
/// Each element: 2 loads + 1 store = 12 bytes, 2 FLOPs (multiply + add).
/// AI = 2/12 ≈ 0.167 FLOP/byte.  Still memory-bound.
__global__ void saxpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

/// **Reduction** — memory-bound with sequential dependency.
///
/// Reads N elements (4N bytes), produces 1 output (4 bytes).
/// N-1 FLOPs (additions).  AI ≈ (N-1)/(4N) → ~0.25 FLOP/byte for large N.
/// Still on the memory-bound side, limited by global memory read bandwidth.
__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? in[i] : 0.0F;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

/// **Heavy compute** — artificial compute-bound kernel.
///
/// Each thread performs `ITERS` iterations of a fused multiply-add chain
/// per element, creating a very high arithmetic intensity.
///
/// Memory: 1 load + 1 store = 8 bytes per element.
/// Compute: ITERS × 2 FLOPs (one multiply + one add).
/// AI = (ITERS × 2) / 8.  With ITERS = 256, AI = 64 FLOP/byte.
/// → **Compute-bound** on virtually any GPU.
constexpr int COMPUTE_ITERS = 256;  ///< 512 FLOP per element → AI = 64.

__global__ void heavy_compute_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    // Iterate many FMAs so the kernel is clearly compute-bound.
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
      val = val * 0.9999F + 0.0001F;
    }
    y[i] = val;
  }
}

/// @}

// =============================================================================
// Benchmark runner
// =============================================================================

/// Run a kernel `n_trials` times (after `n_warmup` warm-ups) and collect stats.
///
/// @tparam KernelFunc  Callable that launches the kernel (no args).
/// @param n_warmup     Number of warm-up iterations (not timed).
/// @param n_trials     Number of timed iterations.
/// @param func         Lambda or function that launches the kernel.
/// @return BenchmarkResult with timing stats.
template <typename KernelFunc>
static BenchmarkResult benchmark_kernel(int n_warmup, int n_trials, KernelFunc func) {
  // Warm-up: let the GPU boost clocks and populate caches.
  for (int i = 0; i < n_warmup; ++i) {
    func();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  std::vector<float> times;
  times.reserve(static_cast<size_t>(n_trials));

  for (int i = 0; i < n_trials; ++i) {
    timer.start();
    func();
    timer.stop();
    times.push_back(timer.elapsed_ms());
  }

  return compute_stats(times);
}

// =============================================================================
// Main — demonstrate benchmarking workflow
// =============================================================================

int main() {
  std::printf("=== Lesson 22: Performance Measurement & Benchmarking ===\n\n");

  // --- GPU info ---
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  float peak_bw = theoretical_peak_bw_gb_s();
  float peak_gf = estimated_peak_gflops();
  float ridge = roofline_ridge_point();

  std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("Theoretical peak memory BW: %.1f GB/s\n", peak_bw);
  std::printf("Estimated peak FP32 GFLOPS: %.0f\n", peak_gf);
  std::printf("Roofline ridge point:       %.2f FLOP/byte\n\n", ridge);

  // --- Allocate data ---
  constexpr int N = 1 << 22;  // 4M elements
  constexpr size_t BYTES = static_cast<size_t>(N) * sizeof(float);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, BYTES));
  CUDA_CHECK(cudaMalloc(&d_b, BYTES));
  CUDA_CHECK(cudaMalloc(&d_c, BYTES));

  // Fill with some data.
  std::vector<float> h_data(N);
  for (int i = 0; i < N; ++i) h_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.001F;
  CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_data.data(), BYTES, cudaMemcpyHostToDevice));

  int blk = 256;
  int grid = (N + blk - 1) / blk;
  constexpr int WARMUP = 5;
  constexpr int TRIALS = 20;

  // -----------------------------------------------------------------------
  // Benchmark 1: Vector Add  (memory-bound)
  // -----------------------------------------------------------------------
  {
    auto result = benchmark_kernel(WARMUP, TRIALS,
                                   [&]() { vector_add_kernel<<<grid, blk>>>(d_a, d_b, d_c, N); });
    // 2 reads + 1 write = 3 × N × 4 bytes.
    size_t total_bytes = 3UL * N * sizeof(float);
    size_t total_flops = static_cast<size_t>(N);  // 1 add per element.
    float ai = static_cast<float>(total_flops) / static_cast<float>(total_bytes);

    std::printf("--- Vector Add (N = %d) ---\n", N);
    std::printf("  Median: %.3f ms   Min: %.3f ms   Max: %.3f ms\n", result.median_ms,
                result.min_ms, result.max_ms);
    std::printf("  Effective bandwidth: %.1f GB/s  (%.1f%% of peak)\n",
                result.bandwidth_gb_s(total_bytes),
                result.bandwidth_gb_s(total_bytes) / peak_bw * 100.0F);
    std::printf("  Arithmetic intensity: %.3f FLOP/byte  →  %s\n", ai,
                ai < ridge ? "MEMORY-BOUND" : "COMPUTE-BOUND");
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // Benchmark 2: Saxpy (memory-bound, 2 FLOP/element)
  // -----------------------------------------------------------------------
  {
    CUDA_CHECK(cudaMemcpy(d_c, h_data.data(), BYTES, cudaMemcpyHostToDevice));
    auto result =
        benchmark_kernel(WARMUP, TRIALS, [&]() { saxpy_kernel<<<grid, blk>>>(2.0F, d_a, d_c, N); });
    size_t total_bytes = 3UL * N * sizeof(float);  // 2 loads (x, y) + 1 store (y).
    size_t total_flops = 2UL * N;                  // multiply + add.
    float ai = static_cast<float>(total_flops) / static_cast<float>(total_bytes);

    std::printf("--- Saxpy (N = %d) ---\n", N);
    std::printf("  Median: %.3f ms   Min: %.3f ms   Max: %.3f ms\n", result.median_ms,
                result.min_ms, result.max_ms);
    std::printf("  Effective bandwidth: %.1f GB/s  (%.1f%% of peak)\n",
                result.bandwidth_gb_s(total_bytes),
                result.bandwidth_gb_s(total_bytes) / peak_bw * 100.0F);
    std::printf("  Arithmetic intensity: %.3f FLOP/byte  →  %s\n", ai,
                ai < ridge ? "MEMORY-BOUND" : "COMPUTE-BOUND");
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // Benchmark 3: Reduction (memory-bound)
  // -----------------------------------------------------------------------
  {
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    auto result = benchmark_kernel(WARMUP, TRIALS, [&]() {
      CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
      reduce_sum_kernel<<<grid, blk, static_cast<size_t>(blk) * sizeof(float)>>>(d_a, d_sum, N);
    });
    size_t total_bytes = static_cast<size_t>(N) * sizeof(float);  // N reads.
    size_t total_flops = static_cast<size_t>(N) - 1;              // N-1 adds.
    float ai = static_cast<float>(total_flops) / static_cast<float>(total_bytes);

    std::printf("--- Reduction (N = %d) ---\n", N);
    std::printf("  Median: %.3f ms   Min: %.3f ms   Max: %.3f ms\n", result.median_ms,
                result.min_ms, result.max_ms);
    std::printf("  Effective bandwidth: %.1f GB/s  (%.1f%% of peak)\n",
                result.bandwidth_gb_s(total_bytes),
                result.bandwidth_gb_s(total_bytes) / peak_bw * 100.0F);
    std::printf("  Arithmetic intensity: %.3f FLOP/byte  →  %s\n", ai,
                ai < ridge ? "MEMORY-BOUND" : "COMPUTE-BOUND");
    std::printf("\n");

    CUDA_CHECK(cudaFree(d_sum));
  }

  // -----------------------------------------------------------------------
  // Benchmark 4: Heavy compute (compute-bound)
  // -----------------------------------------------------------------------
  {
    auto result = benchmark_kernel(WARMUP, TRIALS,
                                   [&]() { heavy_compute_kernel<<<grid, blk>>>(d_a, d_c, N); });
    size_t total_bytes = 2UL * N * sizeof(float);                       // 1 load + 1 store.
    size_t total_flops = 2UL * COMPUTE_ITERS * static_cast<size_t>(N);  // ITERS FMAs × 2.
    float ai = static_cast<float>(total_flops) / static_cast<float>(total_bytes);

    std::printf("--- Heavy Compute, %d FMA/elem (N = %d) ---\n", COMPUTE_ITERS, N);
    std::printf("  Median: %.3f ms   Min: %.3f ms   Max: %.3f ms\n", result.median_ms,
                result.min_ms, result.max_ms);
    std::printf("  GFLOPS: %.1f  (%.1f%% of peak)\n", result.gflops(total_flops),
                result.gflops(total_flops) / peak_gf * 100.0F);
    std::printf("  Arithmetic intensity: %.2f FLOP/byte  →  %s\n", ai,
                ai < ridge ? "MEMORY-BOUND" : "COMPUTE-BOUND");
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // Roofline summary
  // -----------------------------------------------------------------------
  std::printf("=== Roofline Summary ===\n");
  std::printf("  Ridge point: %.2f FLOP/byte\n", ridge);
  std::printf("  Kernels with AI < %.2f are memory-bound (focus on BW).\n", ridge);
  std::printf("  Kernels with AI > %.2f are compute-bound (focus on FLOPS).\n", ridge);
  std::printf("\n  Tip: Use `nsys profile ./22_performance_measurement` for\n");
  std::printf("  timeline profiling, and `ncu ./22_performance_measurement`\n");
  std::printf("  for per-kernel roofline analysis.\n");

  // Cleanup.
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
