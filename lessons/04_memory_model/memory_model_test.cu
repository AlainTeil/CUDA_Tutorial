/**
 * @file memory_model_test.cu
 * @brief Unit tests for Lesson 04 — CUDA Memory Model.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

// ============================================================================
// Part A — Global Memory: vector add
// ============================================================================

__global__ void vector_add_global(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

class VectorAddTest : public ::testing::TestWithParam<int> {};

TEST_P(VectorAddTest, CorrectSum) {
  int n = GetParam();
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> ha(static_cast<size_t>(n)), hb(static_cast<size_t>(n)),
      hc(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    ha[static_cast<size_t>(i)] = static_cast<float>(i);
    hb[static_cast<size_t>(i)] = static_cast<float>(i) * 0.5F;
  }

  float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dc, bytes));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_add_global<<<blocks, threads>>>(da, db, dc, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(hc[static_cast<size_t>(i)], static_cast<float>(i) * 1.5F, 1e-5F);
  }

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
}

INSTANTIATE_TEST_SUITE_P(Sizes, VectorAddTest, ::testing::Values(1, 31, 256, 1000, 4096));

// ============================================================================
// Part B — Constant Memory: polynomial eval
// ============================================================================

__constant__ float kCoeffs[4];

__global__ void poly_eval_constant(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float xi = x[idx];
    y[idx] = kCoeffs[0] + kCoeffs[1] * xi + kCoeffs[2] * xi * xi + kCoeffs[3] * xi * xi * xi;
  }
}

TEST(ConstantMemoryTest, PolynomialEval) {
  constexpr int kN = 512;
  float coeffs[4] = {1.0F, -1.0F, 0.5F, 0.25F};
  CUDA_CHECK(cudaMemcpyToSymbol(kCoeffs, coeffs, sizeof(coeffs)));

  std::vector<float> hx(kN), hy(kN);
  for (int i = 0; i < kN; ++i) hx[static_cast<size_t>(i)] = static_cast<float>(i) * 0.02F;

  float *dx, *dy;
  size_t bytes = kN * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dy, bytes));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

  poly_eval_constant<<<(kN + 255) / 256, 256>>>(dx, dy, kN);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; ++i) {
    float xi = hx[static_cast<size_t>(i)];
    float expected = coeffs[0] + coeffs[1] * xi + coeffs[2] * xi * xi + coeffs[3] * xi * xi * xi;
    EXPECT_NEAR(hy[static_cast<size_t>(i)], expected, 1e-4F) << "at index " << i;
  }

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
}

// ============================================================================
// Part C — Shared Memory: 1-D stencil
// ============================================================================

constexpr int kStencilBlock = 256;

__global__ void stencil_shared(const float* in, float* out, int n) {
  __shared__ float tile[kStencilBlock + 2];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int lidx = threadIdx.x + 1;

  if (gidx < n) tile[lidx] = in[gidx];
  if (threadIdx.x == 0) {
    tile[0] = (gidx > 0) ? in[gidx - 1] : 0.0F;
  }
  if (threadIdx.x == blockDim.x - 1 || gidx == n - 1) {
    tile[lidx + 1] = (gidx + 1 < n) ? in[gidx + 1] : 0.0F;
  }
  __syncthreads();

  if (gidx < n) {
    out[gidx] = (tile[lidx - 1] + tile[lidx] + tile[lidx + 1]) / 3.0F;
  }
}

class StencilTest : public ::testing::TestWithParam<int> {};

TEST_P(StencilTest, ThreePointAverage) {
  int n = GetParam();
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> hin(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) hin[static_cast<size_t>(i)] = static_cast<float>(i);

  float *din, *dout;
  CUDA_CHECK(cudaMalloc(&din, bytes));
  CUDA_CHECK(cudaMalloc(&dout, bytes));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

  stencil_shared<<<(n + kStencilBlock - 1) / kStencilBlock, kStencilBlock>>>(din, dout, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hout(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    float left = (i > 0) ? hin[static_cast<size_t>(i - 1)] : 0.0F;
    float right = (i < n - 1) ? hin[static_cast<size_t>(i + 1)] : 0.0F;
    float expected = (left + hin[static_cast<size_t>(i)] + right) / 3.0F;
    EXPECT_NEAR(hout[static_cast<size_t>(i)], expected, 1e-4F) << "at index " << i;
  }

  CUDA_CHECK(cudaFree(din));
  CUDA_CHECK(cudaFree(dout));
}

INSTANTIATE_TEST_SUITE_P(Sizes, StencilTest, ::testing::Values(1, 3, 255, 256, 257, 1024));
