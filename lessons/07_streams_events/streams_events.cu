/**
 * @file streams_events.cu
 * @brief Lesson 07 — CUDA Streams and Events.
 *
 * ## Streams
 *
 * A **stream** is a sequence of operations (copies, kernels) that execute
 * in order.  Operations in *different* streams may execute **concurrently**.
 * By default, all operations go to stream 0 (the "default stream"), which
 * serialises everything.  Creating extra streams lets the GPU overlap:
 *
 *   - H→D copy in stream 1  |  kernel in stream 2  |  D→H copy in stream 3
 *
 * This is called **latency hiding** and can dramatically improve throughput
 * when the workload is small enough that the GPU has idle resources.
 *
 * ### Requirements for overlap
 * - Host memory must be **pinned** (`cudaMallocHost`).  Pageable memory
 *   forces the driver into a synchronous staging path.
 * - The GPU must have a free copy engine.  Most GPUs have at least two
 *   (one for H→D, one for D→H), so a kernel + one copy can always overlap.
 *
 * ## Events
 *
 * A **CUDA event** is a timestamp marker inserted into a stream.  The pair
 * `cudaEventRecord(start)` / `cudaEventRecord(stop)` + `cudaEventElapsedTime`
 * gives sub-millisecond GPU-side timing without any host-side noise.
 *
 * Events are the foundation of the `GpuTimer` class in Lesson 22.
 *
 * ## This demo
 *
 * We split a large array into `kNStreams` chunks.  Each chunk is
 * independently transferred H→D, processed, and transferred D→H in its own
 * stream.  We compare total wall-clock time (measured by events) of the
 * serial path vs. the concurrent path.
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
// Kernel: element-wise multiply by a scalar
// ---------------------------------------------------------------------------

__global__ void scale_kernel(float* data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] *= factor;
}

// ---------------------------------------------------------------------------
// Serial execution (default stream)
// ---------------------------------------------------------------------------

/// All operations go to stream 0.  The H→D copy, kernel, and D→H copy are
/// strictly serialised — each one must finish before the next starts.
/// This is the simplest model but leaves copy engines idle while the kernel
/// runs (and vice versa).
float run_serial(float* h_data, float* d_data, int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
  scale_kernel<<<blocks, threads>>>(d_data, factor, n);
  CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

// ---------------------------------------------------------------------------
// Concurrent execution using multiple streams
// ---------------------------------------------------------------------------

/// Each stream handles one chunk of the array:
///   stream[i]:  H→D copy → kernel → D→H copy
///
/// Because each stream operates on a **disjoint** region of the array,
/// there are no data dependencies between streams and the GPU can overlap
/// copies with computation.  On an RTX 3070, this typically yields a ~1.5×
/// speedup over the serial path for this simple workload.
///
/// Key API: `cudaMemcpyAsync` (non-blocking copy), and the 4th kernel
/// launch parameter `<<<blocks, threads, 0, stream>>>` which assigns the
/// kernel to a specific stream.
float run_concurrent(float* h_data, float* d_data, int n, float factor, int num_streams) {
  int chunk_size = (n + num_streams - 1) / num_streams;

  std::vector<cudaStream_t> streams(static_cast<size_t>(num_streams));
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < num_streams; ++i) {
    int offset = i * chunk_size;
    int size = (offset + chunk_size <= n) ? chunk_size : (n - offset);
    if (size <= 0) break;

    size_t chunk_bytes = static_cast<size_t>(size) * sizeof(float);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                               cudaMemcpyHostToDevice, streams[static_cast<size_t>(i)]));
    scale_kernel<<<blocks, threads, 0, streams[static_cast<size_t>(i)]>>>(d_data + offset, factor,
                                                                          size);
    CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes,
                               cudaMemcpyDeviceToHost, streams[static_cast<size_t>(i)]));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  return ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  constexpr int kN = 1 << 22;  // ~4 M elements
  constexpr float kFactor = 2.0F;
  constexpr int kNStreams = 4;

  size_t bytes = static_cast<size_t>(kN) * sizeof(float);

  // Use pinned memory for async transfers
  float* h_data = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_data, bytes));
  for (int i = 0; i < kN; ++i) h_data[i] = static_cast<float>(i);

  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  // --- Serial ---
  float serial_ms = run_serial(h_data, d_data, kN, kFactor);

  // Reset host data
  for (int i = 0; i < kN; ++i) h_data[i] = static_cast<float>(i);

  // --- Concurrent ---
  float concurrent_ms = run_concurrent(h_data, d_data, kN, kFactor, kNStreams);

  std::printf("Streams benchmark (n = %d, streams = %d):\n", kN, kNStreams);
  std::printf("  Serial     : %.3f ms\n", static_cast<double>(serial_ms));
  std::printf("  Concurrent : %.3f ms\n", static_cast<double>(concurrent_ms));

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFreeHost(h_data));
  return EXIT_SUCCESS;
}
