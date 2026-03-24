/**
 * @file embeddings_test.cu
 * @brief Unit tests for Lesson 28 — Token Embeddings & Positional Encoding.
 *
 * Tests cover the embedding lookup, sinusoidal positional encoding, backward
 * gradient scatter, and cuBLAS linear projection.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    const cublasStatus_t st_ = (call);                                        \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                    \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self‑contained lesson)
// =============================================================================

__global__ void embedding_forward_kernel(const float* __restrict__ table,
                                         const int* __restrict__ ids, float* __restrict__ out,
                                         int total, int D, int V) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  int token_id = ids[row];
  assert(token_id >= 0 && token_id < V && "token ID out of range");
  (void)V;
  out[row * D + col] = table[token_id * D + col];
}

__global__ void sinusoidal_pe_kernel(float* __restrict__ data, int T, int D, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  int pos = row % T;
  float exponent = static_cast<float>((col / 2) * 2) / static_cast<float>(D);
  float freq = 1.0F / powf(10000.0F, exponent);
  float angle = static_cast<float>(pos) * freq;
  float pe = (col % 2 == 0) ? sinf(angle) : cosf(angle);
  data[idx] += pe;
}

__global__ void embedding_backward_kernel(float* __restrict__ d_table_grad,
                                          const int* __restrict__ ids,
                                          const float* __restrict__ dout, int total, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  atomicAdd(&d_table_grad[ids[row] * D + col], dout[row * D + col]);
}

// =============================================================================
// cuBLAS projection helper
// =============================================================================

static void linear_project(cublasHandle_t handle, const float* d_X, const float* d_W, float* d_out,
                           int M, int K, int N) {
  float alpha = 1.0F, beta = 0.0F;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_W, K, d_X, K, &beta,
                           d_out, N));
}

// =============================================================================
// Tests
// =============================================================================

class EmbeddingsTest : public ::testing::Test {
 protected:
  static constexpr int kV = 8;
  static constexpr int kD = 4;
  static constexpr int kT = 3;
  static constexpr int kB = 2;
  static constexpr int kTotal = kB * kT;  // 6

  float* d_table_{nullptr};
  float* d_out_{nullptr};
  int* d_ids_{nullptr};

  void SetUp() override {
    CUDA_CHECK(cudaMalloc(&d_table_, kV * kD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_, kTotal * kD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids_, kTotal * sizeof(int)));
  }
  void TearDown() override {
    if (d_table_) CUDA_CHECK(cudaFree(d_table_));
    if (d_out_) CUDA_CHECK(cudaFree(d_out_));
    if (d_ids_) CUDA_CHECK(cudaFree(d_ids_));
  }
};

// ------------------------------------------------------------------
// Forward: simple gather produces correct rows
// ------------------------------------------------------------------

TEST_F(EmbeddingsTest, ForwardGathersCorrectRows) {
  // Table: row i is filled with float(i)
  std::vector<float> h_table(kV * kD);
  for (int r = 0; r < kV; ++r)
    for (int c = 0; c < kD; ++c) h_table[r * kD + c] = static_cast<float>(r);

  std::vector<int> h_ids = {0, 3, 7, 1, 5, 2};  // kTotal = 6

  CUDA_CHECK(cudaMemcpy(d_table_, h_table.data(), kV * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ids_, h_ids.data(), kTotal * sizeof(int), cudaMemcpyHostToDevice));

  int elems = kTotal * kD;
  int grid = (elems + kBlockSize - 1) / kBlockSize;
  embedding_forward_kernel<<<grid, kBlockSize>>>(d_table_, d_ids_, d_out_, kTotal, kD, kV);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(kTotal * kD);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_, kTotal * kD * sizeof(float), cudaMemcpyDeviceToHost));

  for (int t = 0; t < kTotal; ++t) {
    for (int d = 0; d < kD; ++d) {
      EXPECT_FLOAT_EQ(h_out[t * kD + d], static_cast<float>(h_ids[t]))
          << "row=" << t << " col=" << d;
    }
  }
}

// ------------------------------------------------------------------
// Positional encoding: position 0 adds sin(0)=0 to even, cos(0)=1 to odd
// ------------------------------------------------------------------

TEST_F(EmbeddingsTest, PositionalEncodingPos0) {
  std::vector<float> h_data(kTotal * kD, 0.0F);
  CUDA_CHECK(
      cudaMemcpy(d_out_, h_data.data(), kTotal * kD * sizeof(float), cudaMemcpyHostToDevice));

  int elems = kTotal * kD;
  int grid = (elems + kBlockSize - 1) / kBlockSize;
  sinusoidal_pe_kernel<<<grid, kBlockSize>>>(d_out_, kT, kD, kTotal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_data.data(), d_out_, kTotal * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // Row 0 → position 0:  sin(0)=0 for even cols, cos(0)=1 for odd cols
  for (int d = 0; d < kD; ++d) {
    float expected = (d % 2 == 0) ? 0.0F : 1.0F;
    EXPECT_NEAR(h_data[d], expected, 1e-5F) << "col=" << d;
  }
}

// ------------------------------------------------------------------
// Backward: gradient accumulates for duplicate token IDs
// ------------------------------------------------------------------

TEST_F(EmbeddingsTest, BackwardAccumulatesDuplicates) {
  // All IDs point to token 0 → gradient for token 0 should be kTotal * 1.0
  std::vector<int> h_ids(kTotal, 0);
  std::vector<float> h_dout(kTotal * kD, 1.0F);

  float* d_table_grad;
  float* d_dout;
  CUDA_CHECK(cudaMalloc(&d_table_grad, kV * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, kTotal * kD * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_table_grad, 0, kV * kD * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_ids_, h_ids.data(), kTotal * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_dout, h_dout.data(), kTotal * kD * sizeof(float), cudaMemcpyHostToDevice));

  int elems = kTotal * kD;
  int grid = (elems + kBlockSize - 1) / kBlockSize;
  embedding_backward_kernel<<<grid, kBlockSize>>>(d_table_grad, d_ids_, d_dout, kTotal, kD);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad(kV * kD);
  CUDA_CHECK(
      cudaMemcpy(h_grad.data(), d_table_grad, kV * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // Token 0 grad should be kTotal (6 duplicates × 1.0)
  for (int d = 0; d < kD; ++d) {
    EXPECT_NEAR(h_grad[d], static_cast<float>(kTotal), 1e-4F) << "col=" << d;
  }
  // Token 1 grad should be 0
  for (int d = 0; d < kD; ++d) {
    EXPECT_FLOAT_EQ(h_grad[kD + d], 0.0F) << "col=" << d;
  }

  CUDA_CHECK(cudaFree(d_table_grad));
  CUDA_CHECK(cudaFree(d_dout));
}

// ------------------------------------------------------------------
// cuBLAS projection: identity weight → output equals input subset
// ------------------------------------------------------------------

TEST_F(EmbeddingsTest, CuBLASProjectionIdentity) {
  // X is (kTotal × kD), W is (kD × kD) — identity
  std::vector<float> h_X(kTotal * kD);
  for (int i = 0; i < kTotal * kD; ++i) h_X[i] = static_cast<float>(i);

  std::vector<float> h_W(kD * kD, 0.0F);
  for (int i = 0; i < kD; ++i) h_W[i * kD + i] = 1.0F;

  float *d_X, *d_W, *d_Y;
  CUDA_CHECK(cudaMalloc(&d_X, kTotal * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_W, kD * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, kTotal * kD * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), kTotal * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), kD * kD * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  linear_project(handle, d_X, d_W, d_Y, kTotal, kD, kD);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_Y(kTotal * kD);
  CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, kTotal * kD * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kTotal * kD; ++i) {
    EXPECT_NEAR(h_Y[i], h_X[i], 1e-4F) << "index " << i;
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_Y));
}

// ------------------------------------------------------------------
// PE is periodic across batches: row 0 and row T should match
// ------------------------------------------------------------------

TEST_F(EmbeddingsTest, PositionalEncodingPeriodicAcrossBatches) {
  std::vector<float> h_data(kTotal * kD, 0.0F);
  CUDA_CHECK(
      cudaMemcpy(d_out_, h_data.data(), kTotal * kD * sizeof(float), cudaMemcpyHostToDevice));

  int elems = kTotal * kD;
  int grid = (elems + kBlockSize - 1) / kBlockSize;
  sinusoidal_pe_kernel<<<grid, kBlockSize>>>(d_out_, kT, kD, kTotal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_data.data(), d_out_, kTotal * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // Row 0 (batch 0, pos 0) should match row T (batch 1, pos 0)
  for (int d = 0; d < kD; ++d) {
    EXPECT_NEAR(h_data[d], h_data[kT * kD + d], 1e-5F)
        << "PE should repeat across batches at col=" << d;
  }
}
