/**
 * @file performance_measurement_test.cu
 * @brief Unit tests for Lesson 22 — Performance Measurement & Benchmarking.
 *
 * These tests verify that:
 *   1. The GpuTimer returns positive, plausible elapsed times.
 *   2. The BenchmarkResult statistics are computed correctly.
 *   3. Effective bandwidth and GFLOPS calculations are sane.
 *   4. Theoretical peak helpers produce positive values.
 *   5. Roofline classification works for known kernel profiles.
 *   6. Benchmark kernels produce numerically correct results.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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
// GpuTimer (duplicated for independent compilation)
// =============================================================================

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
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_event, stream)); }
  void stop(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(stop_event, stream)); }
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return ms;
  }
};

// =============================================================================
// BenchmarkResult (duplicated for independent compilation)
// =============================================================================

struct BenchmarkResult {
  float min_ms{};
  float max_ms{};
  float median_ms{};
  float mean_ms{};
  int trials{};

  float bandwidth_gb_s(size_t bytes) const {
    return static_cast<float>(bytes) / (median_ms * 1e6F);
  }
  float gflops(size_t flops) const { return static_cast<float>(flops) / (median_ms * 1e6F); }
};

static BenchmarkResult compute_stats(std::vector<float>& times) {
  std::sort(times.begin(), times.end());
  BenchmarkResult r;
  r.trials = static_cast<int>(times.size());
  r.min_ms = times.front();
  r.max_ms = times.back();
  r.mean_ms = std::accumulate(times.begin(), times.end(), 0.0F) / static_cast<float>(times.size());
  size_t n = times.size();
  if (n % 2 == 0) {
    r.median_ms = (times[n / 2 - 1] + times[n / 2]) / 2.0F;
  } else {
    r.median_ms = times[n / 2];
  }
  return r;
}

template <typename KernelFunc>
static BenchmarkResult benchmark_kernel(int n_warmup, int n_trials, KernelFunc func) {
  for (int i = 0; i < n_warmup; ++i) func();
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
// Theoretical peak helpers (duplicated)
// =============================================================================

static float theoretical_peak_bw_gb_s(int device = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int mem_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
  double bw = static_cast<double>(mem_clock_khz) * 1e3 *
              (static_cast<double>(prop.memoryBusWidth) / 8.0) * 2.0 / 1e9;
  return static_cast<float>(bw);
}

static float estimated_peak_gflops(int device = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int cores_per_sm;
  if (prop.major == 7 && prop.minor == 0) {
    cores_per_sm = 64;
  } else if (prop.major == 7 && prop.minor == 5) {
    cores_per_sm = 64;
  } else if (prop.major == 8 && prop.minor == 0) {
    cores_per_sm = 64;
  } else if (prop.major == 8 && prop.minor == 6) {
    cores_per_sm = 128;
  } else if (prop.major == 8 && prop.minor == 9) {
    cores_per_sm = 128;
  } else if (prop.major == 9 && prop.minor == 0) {
    cores_per_sm = 128;
  } else {
    cores_per_sm = 64;
  }
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  double gflops = static_cast<double>(prop.multiProcessorCount) *
                  static_cast<double>(cores_per_sm) * static_cast<double>(clock_khz) * 1e3 * 2.0 /
                  1e9;
  return static_cast<float>(gflops);
}

static float roofline_ridge_point(int device = 0) {
  return estimated_peak_gflops(device) / theoretical_peak_bw_gb_s(device);
}

// =============================================================================
// Kernels (duplicated)
// =============================================================================

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void saxpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

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

constexpr int COMPUTE_ITERS = 256;

__global__ void heavy_compute_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    for (int iter = 0; iter < COMPUTE_ITERS; ++iter) {
      val = val * 0.9999F + 0.0001F;
    }
    y[i] = val;
  }
}

// =============================================================================
// Tests
// =============================================================================

/// --- 1. GpuTimer returns a positive elapsed time --->
TEST(PerformanceMeasurement, GpuTimerPositive) {
  GpuTimer timer;
  timer.start();
  // Launch a trivial kernel to create measurable work.
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, 256 * sizeof(float)));
  vector_add_kernel<<<1, 256>>>(d_a, d_b, d_c, 256);
  timer.stop();
  float ms = timer.elapsed_ms();
  EXPECT_GT(ms, 0.0F) << "GPU timer should report positive elapsed time";
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

/// --- 2. Compute stats: min, max, median, mean --->
TEST(PerformanceMeasurement, ComputeStatsCorrect) {
  std::vector<float> times = {5.0F, 1.0F, 3.0F, 2.0F, 4.0F};
  auto r = compute_stats(times);
  EXPECT_EQ(r.trials, 5);
  EXPECT_FLOAT_EQ(r.min_ms, 1.0F);
  EXPECT_FLOAT_EQ(r.max_ms, 5.0F);
  EXPECT_FLOAT_EQ(r.median_ms, 3.0F);
  EXPECT_FLOAT_EQ(r.mean_ms, 3.0F);
}

/// --- 3. Compute stats with even number of samples (median averaged) --->
TEST(PerformanceMeasurement, ComputeStatsEvenSamples) {
  std::vector<float> times = {4.0F, 1.0F, 3.0F, 2.0F};
  auto r = compute_stats(times);
  EXPECT_EQ(r.trials, 4);
  EXPECT_FLOAT_EQ(r.min_ms, 1.0F);
  EXPECT_FLOAT_EQ(r.max_ms, 4.0F);
  EXPECT_FLOAT_EQ(r.median_ms, 2.5F);  // Avg of 2 and 3.
  EXPECT_FLOAT_EQ(r.mean_ms, 2.5F);
}

/// --- 4. BenchmarkResult bandwidth computation --->
TEST(PerformanceMeasurement, BandwidthComputation) {
  BenchmarkResult r;
  r.median_ms = 1.0F;  // 1 ms.
  // If we transfer 1 GB in 1 ms → 1000 GB/s.
  size_t bytes = 1'000'000'000UL;  // 1 GB.
  float bw = r.bandwidth_gb_s(bytes);
  // 1e9 / (1e-3 * 1e9) = 1e9 / 1e6 = 1000.
  EXPECT_NEAR(bw, 1000.0F, 0.1F);
}

/// --- 5. BenchmarkResult GFLOPS computation --->
TEST(PerformanceMeasurement, GflopsComputation) {
  BenchmarkResult r;
  r.median_ms = 2.0F;
  size_t flops = 2'000'000'000UL;  // 2 GFLOP.
  float gf = r.gflops(flops);
  // 2e9 / (2e-3 * 1e9) = 2e9 / 2e6 = 1000.
  EXPECT_NEAR(gf, 1000.0F, 0.1F);
}

/// --- 6. Theoretical peak BW is positive and plausible --->
TEST(PerformanceMeasurement, TheoreticalPeakBW) {
  float bw = theoretical_peak_bw_gb_s();
  EXPECT_GT(bw, 10.0F) << "Peak BW should be > 10 GB/s for any modern GPU";
  EXPECT_LT(bw, 10000.0F) << "Peak BW should be < 10 TB/s (sanity check)";
}

/// --- 7. Estimated peak GFLOPS is positive and plausible --->
TEST(PerformanceMeasurement, EstimatedPeakGflops) {
  float gf = estimated_peak_gflops();
  EXPECT_GT(gf, 100.0F) << "Peak GFLOPS should be > 100 for any modern GPU";
  EXPECT_LT(gf, 100000.0F) << "Peak GFLOPS < 100 TFLOPS (sanity)";
}

/// --- 8. Roofline ridge point is positive --->
TEST(PerformanceMeasurement, RooflineRidgePoint) {
  float ridge = roofline_ridge_point();
  EXPECT_GT(ridge, 0.0F);
  EXPECT_LT(ridge, 1000.0F) << "Ridge point should be reasonable";
}

/// --- 9. Vector add produces correct results --->
TEST(PerformanceMeasurement, VectorAddCorrectness) {
  constexpr int N = 1024;
  std::vector<float> h_a(N), h_b(N), h_c(N);
  for (int i = 0; i < N; ++i) {
    h_a[static_cast<size_t>(i)] = static_cast<float>(i);
    h_b[static_cast<size_t>(i)] = static_cast<float>(i) * 2.0F;
  }
  float *d_a, *d_b, *d_c;
  size_t bytes = N * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  vector_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(h_c[static_cast<size_t>(i)], static_cast<float>(i) * 3.0F);
  }
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

/// --- 10. Saxpy produces correct results --->
TEST(PerformanceMeasurement, SaxpyCorrectness) {
  constexpr int N = 512;
  std::vector<float> h_x(N), h_y(N), h_expected(N);
  for (int i = 0; i < N; ++i) {
    h_x[static_cast<size_t>(i)] = static_cast<float>(i);
    h_y[static_cast<size_t>(i)] = 1.0F;
    h_expected[static_cast<size_t>(i)] = 3.0F * static_cast<float>(i) + 1.0F;
  }
  float *d_x, *d_y;
  size_t bytes = N * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  saxpy_kernel<<<(N + 255) / 256, 256>>>(3.0F, d_x, d_y, N);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(h_y[static_cast<size_t>(i)], h_expected[static_cast<size_t>(i)], 1e-4F);
  }
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
}

/// --- 11. Reduction produces correct sum --->
TEST(PerformanceMeasurement, ReductionCorrectness) {
  constexpr int N = 4096;
  std::vector<float> h_in(N);
  float expected = 0.0F;
  for (int i = 0; i < N; ++i) {
    h_in[static_cast<size_t>(i)] = 1.0F;
    expected += 1.0F;
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

  int blk = 256;
  int grid = (N + blk - 1) / blk;
  reduce_sum_kernel<<<grid, blk, static_cast<size_t>(blk) * sizeof(float)>>>(d_in, d_out, N);

  float h_out = 0.0F;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(h_out, expected, 1e-2F);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

/// --- 12. Benchmark runner returns plausible stats --->
TEST(PerformanceMeasurement, BenchmarkRunnerPlausible) {
  constexpr int N = 1 << 18;  // 256K elements.
  float* d_a;
  float* d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  auto result = benchmark_kernel(
      2, 5, [&]() { vector_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_a, d_c, N); });

  EXPECT_EQ(result.trials, 5);
  EXPECT_GT(result.min_ms, 0.0F);
  EXPECT_GE(result.max_ms, result.min_ms);
  EXPECT_GE(result.median_ms, result.min_ms);
  EXPECT_LE(result.median_ms, result.max_ms);
  EXPECT_GT(result.mean_ms, 0.0F);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_c));
}

/// --- 13. Effective bandwidth is within physical limits --->
TEST(PerformanceMeasurement, EffectiveBandwidthPlausible) {
  constexpr int N = 1 << 20;  // 1M elements.
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  auto result = benchmark_kernel(
      3, 10, [&]() { vector_add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N); });

  size_t total_bytes = 3UL * N * sizeof(float);
  float bw = result.bandwidth_gb_s(total_bytes);
  float peak = theoretical_peak_bw_gb_s();

  EXPECT_GT(bw, 1.0F) << "Bandwidth should be > 1 GB/s";
  EXPECT_LT(bw, peak * 1.1F) << "Bandwidth should not exceed peak by >10%";

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

/// --- 14. Roofline classification of known kernels --->
TEST(PerformanceMeasurement, RooflineClassification) {
  float ridge = roofline_ridge_point();

  // Vector add: AI = 1/12 ≈ 0.083 — should be memory-bound.
  float ai_vadd = 1.0F / 12.0F;
  EXPECT_LT(ai_vadd, ridge) << "Vector add should be memory-bound";

  // Heavy compute: AI = (256×2)/8 = 64 — should be compute-bound.
  float ai_heavy = static_cast<float>(COMPUTE_ITERS * 2) / 8.0F;
  EXPECT_GT(ai_heavy, ridge) << "Heavy compute should be compute-bound (ridge = " << ridge << ")";
}
