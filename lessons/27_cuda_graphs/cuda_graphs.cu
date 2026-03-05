/**
 * @file cuda_graphs.cu
 * @brief Lesson 27 — CUDA Graphs.
 *
 * Every CUDA kernel launch has a **CPU-side overhead** of ~5-10 µs.  For
 * pipelines with many small kernels — typical in deep-learning inference
 * (see Lesson 18) — this overhead can exceed the GPU compute time itself.
 *
 * **CUDA Graphs** capture a sequence of GPU operations (kernels, memcpys,
 * etc.) into a **DAG** that can be "replayed" with a single CPU-side call,
 * amortising launch overhead across the entire graph.
 *
 * ## Part 1 — Stream Capture
 *
 * The easiest path: wrap existing kernel launches with
 * `cudaStreamBeginCapture` / `cudaStreamEndCapture`.  The runtime records
 * every operation issued to the stream and builds the graph automatically.
 *
 * ## Part 2 — Explicit Graph Construction
 *
 * For full control, build the graph node by node:
 * - `cudaGraphCreate(&graph, 0)`
 * - `cudaGraphAddKernelNode(&node, graph, deps, nDeps, &params)`
 * - `cudaGraphAddDependencies(graph, from, to, n)`
 * - `cudaGraphInstantiate(&instance, graph)`
 * - `cudaGraphLaunch(instance, stream)`
 *
 * ## Part 3 — Graph Update
 *
 * When you need to change kernel parameters (e.g., input pointer, grid
 * size) without rebuilding the graph, use:
 * - `cudaGraphExecKernelNodeSetParams(instance, node, &new_params)`
 *
 * This is faster than re-capturing when only arguments change.
 *
 * See Lesson 22 for benchmarking techniques, and Lesson 18 for the
 * inference pipeline that motivates kernel launch overhead reduction.
 */

#include <cmath>
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

constexpr int kBlockSize = 256;
constexpr int kN = 4096;

// =============================================================================
// Simple kernel pipeline (mimics a mini inference pipeline)
// =============================================================================

/**
 * @brief Fill array with a constant value.
 *
 * @param out  Output array.
 * @param n    Number of elements.
 * @param val  Value to fill.
 */
__global__ void fill_kernel(float* __restrict__ out, int n, float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

/**
 * @brief Element-wise scale: out[i] = in[i] * scale.
 *
 * @param in     Input array.
 * @param out    Output array.
 * @param n      Number of elements.
 * @param scale  Scaling factor.
 */
__global__ void scale_kernel(const float* in, float* out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * scale;
}

/**
 * @brief Element-wise bias add: data[i] += bias.
 *
 * @param data  Array to modify in place.
 * @param n     Number of elements.
 * @param bias  Bias value to add.
 */
__global__ void add_bias_kernel(float* __restrict__ data, int n, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += bias;
}

/**
 * @brief ReLU activation: data[i] = max(0, data[i]).
 *
 * @param data  Array to apply ReLU on in place.
 * @param n     Number of elements.
 */
__global__ void relu_kernel(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = fmaxf(0.0F, data[i]);
}

// =============================================================================
// Helpers
// =============================================================================

/// @brief Run the 4-kernel pipeline directly (no graph).
static void run_pipeline_direct(float* d_buf, int n, float fill_val, float scale, float bias,
                                cudaStream_t stream) {
  int grid = (n + kBlockSize - 1) / kBlockSize;
  fill_kernel<<<grid, kBlockSize, 0, stream>>>(d_buf, n, fill_val);
  CUDA_CHECK(cudaGetLastError());
  scale_kernel<<<grid, kBlockSize, 0, stream>>>(d_buf, d_buf, n, scale);
  CUDA_CHECK(cudaGetLastError());
  add_bias_kernel<<<grid, kBlockSize, 0, stream>>>(d_buf, n, bias);
  CUDA_CHECK(cudaGetLastError());
  relu_kernel<<<grid, kBlockSize, 0, stream>>>(d_buf, n);
  CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Part 1 — Stream Capture
// =============================================================================

static void demo_stream_capture(float* d_buf, int n) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ---- Capture ----
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf, n, 2.0F, 0.5F, -0.3F, stream);
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t instance;
  CUDA_CHECK(cudaGraphInstantiate(&instance, graph));

  // ---- Launch ----
  CUDA_CHECK(cudaGraphLaunch(instance, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Verify (first element should be max(0, 2*0.5 + (-0.3)) = 0.7)
  float first = 0.0F;
  CUDA_CHECK(cudaMemcpy(&first, d_buf, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Stream capture result[0] = %.4f (expected 0.7000)\n", static_cast<double>(first));

  // ---- Benchmark: graph launch vs. direct ----
  constexpr int kIter = 1000;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Graph
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < kIter; ++i) CUDA_CHECK(cudaGraphLaunch(instance, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float graph_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&graph_ms, start, stop));

  // Direct
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < kIter; ++i) run_pipeline_direct(d_buf, n, 2.0F, 0.5F, -0.3F, stream);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float direct_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&direct_ms, start, stop));

  std::printf("  %d iterations: graph %.3f ms, direct %.3f ms (%.1fx)\n", kIter,
              static_cast<double>(graph_ms), static_cast<double>(direct_ms),
              static_cast<double>(direct_ms / graph_ms));

  CUDA_CHECK(cudaGraphExecDestroy(instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// =============================================================================
// Part 2 — Explicit Graph Construction
// =============================================================================

static void demo_explicit_graph(float* d_buf, int n) {
  int grid = (n + kBlockSize - 1) / kBlockSize;

  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  // Node 1: fill_kernel
  cudaKernelNodeParams fill_params{};
  float fill_val = 2.0F;
  void* fill_args[] = {&d_buf, const_cast<int*>(&n), &fill_val};
  fill_params.func = reinterpret_cast<void*>(fill_kernel);
  fill_params.gridDim = dim3(grid);
  fill_params.blockDim = dim3(kBlockSize);
  fill_params.sharedMemBytes = 0;
  fill_params.kernelParams = fill_args;
  fill_params.extra = nullptr;

  cudaGraphNode_t fill_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&fill_node, graph, nullptr, 0, &fill_params));

  // Node 2: scale_kernel  (depends on fill)
  cudaKernelNodeParams scale_params{};
  float scale_val = 0.5F;
  void* scale_args[] = {&d_buf, &d_buf, const_cast<int*>(&n), &scale_val};
  scale_params.func = reinterpret_cast<void*>(scale_kernel);
  scale_params.gridDim = dim3(grid);
  scale_params.blockDim = dim3(kBlockSize);
  scale_params.sharedMemBytes = 0;
  scale_params.kernelParams = scale_args;
  scale_params.extra = nullptr;

  cudaGraphNode_t scale_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&scale_node, graph, &fill_node, 1, &scale_params));

  // Node 3: add_bias_kernel  (depends on scale)
  cudaKernelNodeParams bias_params{};
  float bias_val = -0.3F;
  void* bias_args[] = {&d_buf, const_cast<int*>(&n), &bias_val};
  bias_params.func = reinterpret_cast<void*>(add_bias_kernel);
  bias_params.gridDim = dim3(grid);
  bias_params.blockDim = dim3(kBlockSize);
  bias_params.sharedMemBytes = 0;
  bias_params.kernelParams = bias_args;
  bias_params.extra = nullptr;

  cudaGraphNode_t bias_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&bias_node, graph, &scale_node, 1, &bias_params));

  // Node 4: relu_kernel  (depends on bias)
  cudaKernelNodeParams relu_params{};
  void* relu_args[] = {&d_buf, const_cast<int*>(&n)};
  relu_params.func = reinterpret_cast<void*>(relu_kernel);
  relu_params.gridDim = dim3(grid);
  relu_params.blockDim = dim3(kBlockSize);
  relu_params.sharedMemBytes = 0;
  relu_params.kernelParams = relu_args;
  relu_params.extra = nullptr;

  cudaGraphNode_t relu_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&relu_node, graph, &bias_node, 1, &relu_params));

  // Instantiate and launch
  cudaGraphExec_t instance;
  CUDA_CHECK(cudaGraphInstantiate(&instance, graph));
  CUDA_CHECK(cudaGraphLaunch(instance, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  float first = 0.0F;
  CUDA_CHECK(cudaMemcpy(&first, d_buf, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Explicit graph result[0] = %.4f (expected 0.7000)\n", static_cast<double>(first));

  CUDA_CHECK(cudaGraphExecDestroy(instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

// =============================================================================
// Part 3 — Graph Update (change fill value without rebuilding)
// =============================================================================

static void demo_graph_update(float* d_buf, int n) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Capture the pipeline with fill_val = 2.0
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf, n, 2.0F, 0.5F, -0.3F, stream);
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t instance;
  CUDA_CHECK(cudaGraphInstantiate(&instance, graph));

  // Run with original args → result = relu(2*0.5 - 0.3) = 0.7
  CUDA_CHECK(cudaGraphLaunch(instance, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float r1 = 0.0F;
  CUDA_CHECK(cudaMemcpy(&r1, d_buf, sizeof(float), cudaMemcpyDeviceToHost));

  // Now re-capture with different fill value = 10.0
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf, n, 10.0F, 0.5F, -0.3F, stream);
  cudaGraph_t graph2;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph2));

  // Update the existing instance with the new graph
  cudaGraphExecUpdateResultInfo updateResult;
  CUDA_CHECK(cudaGraphExecUpdate(instance, graph2, &updateResult));

  CUDA_CHECK(cudaGraphLaunch(instance, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float r2 = 0.0F;
  CUDA_CHECK(cudaMemcpy(&r2, d_buf, sizeof(float), cudaMemcpyDeviceToHost));

  // Result should change: relu(10*0.5 - 0.3) = 4.7
  std::printf("Graph update: before=%.4f, after=%.4f (expected 0.7 → 4.7)\n",
              static_cast<double>(r1), static_cast<double>(r2));

  CUDA_CHECK(cudaGraphExecDestroy(instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaGraphDestroy(graph2));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// =============================================================================
// main
// =============================================================================

int main() {
  float* d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, kN * sizeof(float)));

  std::printf("=== Part 1: Stream Capture ===\n");
  demo_stream_capture(d_buf, kN);

  std::printf("\n=== Part 2: Explicit Graph Construction ===\n");
  demo_explicit_graph(d_buf, kN);

  std::printf("\n=== Part 3: Graph Update ===\n");
  demo_graph_update(d_buf, kN);

  CUDA_CHECK(cudaFree(d_buf));
  return EXIT_SUCCESS;
}
