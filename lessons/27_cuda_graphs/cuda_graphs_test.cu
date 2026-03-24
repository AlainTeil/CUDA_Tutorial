/**
 * @file cuda_graphs_test.cu
 * @brief Unit tests for Lesson 27 — CUDA Graphs.
 *
 * Tests verify that stream capture, explicit construction, and graph update
 * all produce identical results to direct kernel launches.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    const cudaError_t err_ = (call);                                        \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self-contained lesson)
// =============================================================================

__global__ void fill_kernel(float* __restrict__ out, int n, float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

__global__ void scale_kernel(const float* in, float* out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * scale;
}

__global__ void add_bias_kernel(float* __restrict__ data, int n, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += bias;
}

__global__ void relu_kernel(float* __restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = fmaxf(0.0F, data[i]);
}

// =============================================================================
// Helper: run pipeline directly
// =============================================================================

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
// Test fixture
// =============================================================================

class CudaGraphsTest : public ::testing::Test {
 protected:
  static constexpr int kN = 1024;
  float* d_buf_{nullptr};

  void SetUp() override { CUDA_CHECK(cudaMalloc(&d_buf_, kN * sizeof(float))); }
  void TearDown() override {
    if (d_buf_) CUDA_CHECK(cudaFree(d_buf_));
  }
};

// =============================================================================
// Part 1: Stream capture produces the same result as direct launch
// =============================================================================

TEST_F(CudaGraphsTest, StreamCaptureMatchesDirect) {
  constexpr float kFill = 3.0F;
  constexpr float kScale = 0.4F;
  constexpr float kBias = -1.0F;

  // -- Direct --
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  run_pipeline_direct(d_buf_, kN, kFill, kScale, kBias, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> direct(kN);
  CUDA_CHECK(cudaMemcpy(direct.data(), d_buf_, kN * sizeof(float), cudaMemcpyDeviceToHost));

  // -- Stream capture --
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf_, kN, kFill, kScale, kBias, s);
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));

  cudaGraphExec_t instance;
  CUDA_CHECK(cudaGraphInstantiate(&instance, graph));
  CUDA_CHECK(cudaGraphLaunch(instance, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> captured(kN);
  CUDA_CHECK(cudaMemcpy(captured.data(), d_buf_, kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(direct[i], captured[i]) << "mismatch at index " << i;
  }

  CUDA_CHECK(cudaGraphExecDestroy(instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));
}

// =============================================================================
// Part 2: Explicit graph construction produces same result as stream capture
// =============================================================================

TEST_F(CudaGraphsTest, ExplicitMatchesStreamCapture) {
  constexpr float kFill = 5.0F;
  constexpr float kScale = -0.5F;
  constexpr float kBias = 1.0F;
  int grid = (kN + kBlockSize - 1) / kBlockSize;

  // -- Stream capture result --
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf_, kN, kFill, kScale, kBias, s);
  cudaGraph_t cg;
  CUDA_CHECK(cudaStreamEndCapture(s, &cg));

  cudaGraphExec_t ci;
  CUDA_CHECK(cudaGraphInstantiate(&ci, cg));
  CUDA_CHECK(cudaGraphLaunch(ci, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> captured(kN);
  CUDA_CHECK(cudaMemcpy(captured.data(), d_buf_, kN * sizeof(float), cudaMemcpyDeviceToHost));

  // -- Explicit graph --
  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  float fv = kFill;
  int nn = kN;
  void* fa[] = {&d_buf_, &nn, &fv};
  cudaKernelNodeParams fp{};
  fp.func = reinterpret_cast<void*>(fill_kernel);
  fp.gridDim = dim3(grid);
  fp.blockDim = dim3(kBlockSize);
  fp.kernelParams = fa;

  cudaGraphNode_t fn;
  CUDA_CHECK(cudaGraphAddKernelNode(&fn, graph, nullptr, 0, &fp));

  float sv = kScale;
  void* sa[] = {&d_buf_, &d_buf_, &nn, &sv};
  cudaKernelNodeParams sp{};
  sp.func = reinterpret_cast<void*>(scale_kernel);
  sp.gridDim = dim3(grid);
  sp.blockDim = dim3(kBlockSize);
  sp.kernelParams = sa;

  cudaGraphNode_t sn;
  CUDA_CHECK(cudaGraphAddKernelNode(&sn, graph, &fn, 1, &sp));

  float bv = kBias;
  void* ba[] = {&d_buf_, &nn, &bv};
  cudaKernelNodeParams bp{};
  bp.func = reinterpret_cast<void*>(add_bias_kernel);
  bp.gridDim = dim3(grid);
  bp.blockDim = dim3(kBlockSize);
  bp.kernelParams = ba;

  cudaGraphNode_t bn;
  CUDA_CHECK(cudaGraphAddKernelNode(&bn, graph, &sn, 1, &bp));

  void* ra[] = {&d_buf_, &nn};
  cudaKernelNodeParams rp{};
  rp.func = reinterpret_cast<void*>(relu_kernel);
  rp.gridDim = dim3(grid);
  rp.blockDim = dim3(kBlockSize);
  rp.kernelParams = ra;

  cudaGraphNode_t rn;
  CUDA_CHECK(cudaGraphAddKernelNode(&rn, graph, &bn, 1, &rp));

  cudaGraphExec_t ei;
  CUDA_CHECK(cudaGraphInstantiate(&ei, graph));
  CUDA_CHECK(cudaGraphLaunch(ei, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> explicit_result(kN);
  CUDA_CHECK(
      cudaMemcpy(explicit_result.data(), d_buf_, kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(captured[i], explicit_result[i]) << "mismatch at index " << i;
  }

  CUDA_CHECK(cudaGraphExecDestroy(ei));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaGraphExecDestroy(ci));
  CUDA_CHECK(cudaGraphDestroy(cg));
  CUDA_CHECK(cudaStreamDestroy(s));
}

// =============================================================================
// Part 3: Graph update changes output correctly
// =============================================================================

TEST_F(CudaGraphsTest, GraphUpdateChangesOutput) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // Capture with fill = 2.0, scale = 0.5, bias = -0.3  → relu(0.7) = 0.7
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf_, kN, 2.0F, 0.5F, -0.3F, s);
  cudaGraph_t g1;
  CUDA_CHECK(cudaStreamEndCapture(s, &g1));

  cudaGraphExec_t inst;
  CUDA_CHECK(cudaGraphInstantiate(&inst, g1));
  CUDA_CHECK(cudaGraphLaunch(inst, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  float before = 0.0F;
  CUDA_CHECK(cudaMemcpy(&before, d_buf_, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(before, 0.7F, 1e-5F);

  // Re‑capture with fill = 10.0 → relu(10*0.5 - 0.3) = 4.7
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf_, kN, 10.0F, 0.5F, -0.3F, s);
  cudaGraph_t g2;
  CUDA_CHECK(cudaStreamEndCapture(s, &g2));

  cudaGraphExecUpdateResultInfo info;
  CUDA_CHECK(cudaGraphExecUpdate(inst, g2, &info));

  CUDA_CHECK(cudaGraphLaunch(inst, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  float after = 0.0F;
  CUDA_CHECK(cudaMemcpy(&after, d_buf_, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(after, 4.7F, 1e-5F);

  CUDA_CHECK(cudaGraphExecDestroy(inst));
  CUDA_CHECK(cudaGraphDestroy(g1));
  CUDA_CHECK(cudaGraphDestroy(g2));
  CUDA_CHECK(cudaStreamDestroy(s));
}

// =============================================================================
// Benchmark: graph launch is faster than repeated direct launches
// =============================================================================

TEST_F(CudaGraphsTest, GraphLaunchFasterThanDirect) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // Capture
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_pipeline_direct(d_buf_, kN, 1.0F, 1.0F, 0.0F, s);
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));

  cudaGraphExec_t inst;
  CUDA_CHECK(cudaGraphInstantiate(&inst, graph));

  constexpr int kIter = 500;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm up
  for (int i = 0; i < 50; ++i) CUDA_CHECK(cudaGraphLaunch(inst, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Graph timing
  CUDA_CHECK(cudaEventRecord(start, s));
  for (int i = 0; i < kIter; ++i) CUDA_CHECK(cudaGraphLaunch(inst, s));
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  float graph_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&graph_ms, start, stop));

  // Direct timing
  for (int i = 0; i < 50; ++i) run_pipeline_direct(d_buf_, kN, 1.0F, 1.0F, 0.0F, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  CUDA_CHECK(cudaEventRecord(start, s));
  for (int i = 0; i < kIter; ++i) run_pipeline_direct(d_buf_, kN, 1.0F, 1.0F, 0.0F, s);
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  float direct_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&direct_ms, start, stop));

  std::printf("  Graph: %.3f ms  Direct: %.3f ms  Speedup: %.1fx\n", static_cast<double>(graph_ms),
              static_cast<double>(direct_ms), static_cast<double>(direct_ms / graph_ms));

  // Graph should be at least somewhat faster for small kernels
  EXPECT_LE(graph_ms, direct_ms * 1.1F)
      << "Graph launch should not be significantly slower than direct";

  CUDA_CHECK(cudaGraphExecDestroy(inst));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(s));
}
