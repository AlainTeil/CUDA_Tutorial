/**
 * @file memory_transfers.cu
 * @brief Lesson 06 — Pinned (page-locked) and Unified (managed) memory.
 *
 * Data must travel across the **PCIe bus** (or NVLink) between host RAM and
 * GPU VRAM.  How you allocate host memory has a big impact on transfer speed.
 *
 * ## Three strategies compared here
 *
 * | Strategy | Host allocation | Transfer | Notes |
 * |----------|----------------|----------|-------|
 * | Pageable | `new` / `std::vector` | `cudaMemcpy` | OS may page the data out; driver must stage
 * through a pinned bounce buffer → extra copy | | Pinned   | `cudaMallocHost` | `cudaMemcpy` |
 * DMA-capable; GPU reads directly; ~2× faster transfers | | Unified  | `cudaMallocManaged` | *none*
 * | Driver migrates pages on demand; simplest code, variable perf |
 *
 * ## Why pinned memory is faster
 *
 * The GPU’s DMA engine can only read from **physical** (non-paged) host
 * addresses.  With pageable memory, the CUDA driver must first copy the data
 * into an internal pinned staging buffer, then DMA from there — doubling the
 * work.  `cudaMallocHost` pins the pages up front, eliminating the extra copy.
 *
 * ## Unified memory
 *
 * `cudaMallocManaged` returns a pointer valid on both host and device.  The
 * driver migrates pages automatically.  This is the **simplest** model but
 * can be slower if the access pattern triggers many page faults.  Best for
 * prototyping; production code often prefers explicit transfers with pinned
 * memory + streams (Lesson 07).
 *
 * CUDA events are used to measure transfer + kernel time for each path.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Kernel: scale every element by a constant
// ---------------------------------------------------------------------------

__global__ void scale_kernel(float* data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= factor;
  }
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

struct TimedResult {
  float elapsed_ms;
};

// ---------------------------------------------------------------------------
// Path 1 — Pageable host memory
// ---------------------------------------------------------------------------

TimedResult run_pageable(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // Pageable host allocation
  std::vector<float> h_data(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) h_data[static_cast<size_t>(i)] = static_cast<float>(i);

  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
  scale_kernel<<<(n + 255) / 256, 256>>>(d_data, factor, n);
  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));

  return {ms};
}

// ---------------------------------------------------------------------------
// Path 2 — Pinned host memory
// ---------------------------------------------------------------------------

TimedResult run_pinned(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float* h_data = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_data, bytes));
  for (int i = 0; i < n; ++i) h_data[i] = static_cast<float>(i);

  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
  scale_kernel<<<(n + 255) / 256, 256>>>(d_data, factor, n);
  CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFreeHost(h_data));

  return {ms};
}

// ---------------------------------------------------------------------------
// Path 3 — Unified (managed) memory
// ---------------------------------------------------------------------------

TimedResult run_unified(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float* data = nullptr;
  CUDA_CHECK(cudaMallocManaged(&data, bytes));
  for (int i = 0; i < n; ++i) data[i] = static_cast<float>(i);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  scale_kernel<<<(n + 255) / 256, 256>>>(data, factor, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(data));

  return {ms};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  constexpr int kN = 1 << 22;  // ~4 M elements ≈ 16 MB
  constexpr float kFactor = 2.5F;

  auto r1 = run_pageable(kN, kFactor);
  auto r2 = run_pinned(kN, kFactor);
  auto r3 = run_unified(kN, kFactor);

  std::printf("Memory transfer benchmark (n = %d):\n", kN);
  std::printf("  Pageable : %.3f ms\n", static_cast<double>(r1.elapsed_ms));
  std::printf("  Pinned   : %.3f ms\n", static_cast<double>(r2.elapsed_ms));
  std::printf("  Unified  : %.3f ms\n", static_cast<double>(r3.elapsed_ms));

  return EXIT_SUCCESS;
}
