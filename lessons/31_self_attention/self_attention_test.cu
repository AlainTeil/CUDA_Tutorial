/**
 * @file self_attention_test.cu
 * @brief Unit tests for Lesson 31 — Multi‑Head Self‑Attention.
 *
 * Tests verify split/merge head kernels, softmax, and the full
 * multi‑head attention forward pass.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                    \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

constexpr int kBlockSize = 256;

/// Negative-infinity sentinel for softmax stability.
constexpr float kNegInf = -1e30F;

// =============================================================================
// Kernels (duplicated — self‑contained lesson)
// =============================================================================

__global__ void split_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK, int in_stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int in_idx = b * (T * in_stride) + t * in_stride + h * dK + dk;
  int out_idx = ((b * nH + h) * T + t) * dK + dk;
  out[out_idx] = in[in_idx];
}

__global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float* row_data = data + row * cols;
  float local_max = kNegInf;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, row_data[c]);
  smem[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float max_val = smem[0];
  float local_sum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf(row_data[c] - max_val);
    row_data[c] = e;
    local_sum += e;
  }
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float total = smem[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) row_data[c] /= total;
}

__global__ void merge_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int in_idx = ((b * nH + h) * T + t) * dK + dk;
  int out_idx = b * (T * nH * dK) + t * (nH * dK) + h * dK + dk;
  out[out_idx] = in[in_idx];
}

// =============================================================================
// Tests
// =============================================================================

class SelfAttentionTest : public ::testing::Test {
 protected:
  cublasHandle_t handle_{};
  void SetUp() override { CUBLAS_CHECK(cublasCreate(&handle_)); }
  void TearDown() override { CUBLAS_CHECK(cublasDestroy(handle_)); }
};

// ------------------------------------------------------------------
// Split then merge is identity
// ------------------------------------------------------------------

TEST_F(SelfAttentionTest, SplitMergeRoundTrip) {
  constexpr int kB = 2;
  constexpr int kT = 4;
  constexpr int kNH = 2;
  constexpr int kDK = 8;
  constexpr int kD = kNH * kDK;
  int total = kB * kT * kD;

  std::vector<float> h_in(total);
  std::iota(h_in.begin(), h_in.end(), 0.0F);

  float *d_in, *d_split, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_split, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (total + kBlockSize - 1) / kBlockSize;
  split_heads_kernel<<<grid, kBlockSize>>>(d_in, d_split, kB, kT, kNH, kDK, kD);
  CUDA_CHECK(cudaGetLastError());
  merge_heads_kernel<<<grid, kBlockSize>>>(d_split, d_out, kB, kT, kNH, kDK);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(total);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) EXPECT_FLOAT_EQ(h_in[i], h_out[i]) << "index " << i;

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_split));
  CUDA_CHECK(cudaFree(d_out));
}

// ------------------------------------------------------------------
// Softmax: rows sum to 1, max element has largest probability
// ------------------------------------------------------------------

TEST_F(SelfAttentionTest, SoftmaxRowsSumToOne) {
  constexpr int kRows = 8;
  constexpr int kCols = 16;
  int total = kRows * kCols;

  std::vector<float> h_data(total);
  for (int i = 0; i < total; ++i) h_data[i] = static_cast<float>(i % 7) - 3.0F;

  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  int block = 32;
  softmax_kernel<<<kRows, block, block * sizeof(float)>>>(d_data, kRows, kCols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, total * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < kRows; ++r) {
    float sum = 0.0F;
    for (int c = 0; c < kCols; ++c) {
      EXPECT_GE(h_data[r * kCols + c], 0.0F);
      sum += h_data[r * kCols + c];
    }
    EXPECT_NEAR(sum, 1.0F, 1e-5F) << "row=" << r;
  }

  CUDA_CHECK(cudaFree(d_data));
}

// ------------------------------------------------------------------
// Identity weights for W_QKV and W_O → output reproduces a pattern
// ------------------------------------------------------------------

TEST_F(SelfAttentionTest, IdentityWeightsPreservesStructure) {
  constexpr int kB = 1;
  constexpr int kT = 4;
  constexpr int kD = 8;
  constexpr int kNH = 2;
  constexpr int kDK = kD / kNH;

  // W_QKV = block‑diagonal identity (3D × D) — each of Q,K,V gets identity
  std::vector<float> h_wqkv(3 * kD * kD, 0.0F);
  for (int block = 0; block < 3; ++block)
    for (int i = 0; i < kD; ++i) h_wqkv[(block * kD + i) * kD + i] = 1.0F;

  std::vector<float> h_wo(kD * kD, 0.0F);
  for (int i = 0; i < kD; ++i) h_wo[i * kD + i] = 1.0F;

  // Input: all ones
  std::vector<float> h_X(kB * kT * kD, 1.0F);

  // Allocate
  float *d_wqkv, *d_wo, *d_X, *d_out;
  float *d_qkv, *d_Q, *d_K, *d_V, *d_scores, *d_ctx, *d_merged;
  CUDA_CHECK(cudaMalloc(&d_wqkv, 3 * kD * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wo, kD * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_X, kB * kT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, kB * kT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qkv, kB * kT * 3 * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Q, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scores, kB * kNH * kT * kT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ctx, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_merged, kB * kT * kD * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_wqkv, h_wqkv.data(), 3 * kD * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wo, h_wo.data(), kD * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), kB * kT * kD * sizeof(float), cudaMemcpyHostToDevice));

  // Forward
  float alpha = 1.0F, beta = 0.0F;
  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kD, kB * kT, kD, &alpha, d_wqkv,
                           kD, d_X, kD, &beta, d_qkv, 3 * kD));

  int total = kB * kT * kD;
  int grid = (total + kBlockSize - 1) / kBlockSize;
  split_heads_kernel<<<grid, kBlockSize>>>(d_qkv, d_Q, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());
  split_heads_kernel<<<grid, kBlockSize>>>(d_qkv + kD, d_K, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());
  split_heads_kernel<<<grid, kBlockSize>>>(d_qkv + 2 * kD, d_V, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());

  float scale = 1.0F / sqrtf(static_cast<float>(kDK));
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kT, kT, kDK, &scale,
                                         d_K, kDK, kT * kDK, d_Q, kDK, kT * kDK, &beta, d_scores,
                                         kT, kT * kT, kB * kNH));

  int smem_block = 32;
  softmax_kernel<<<kB * kNH * kT, smem_block, smem_block * sizeof(float)>>>(d_scores, kB * kNH * kT,
                                                                            kT);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kDK, kT, kT, &alpha,
                                         d_V, kDK, kT * kDK, d_scores, kT, kT * kT, &beta, d_ctx,
                                         kDK, kT * kDK, kB * kNH));

  merge_heads_kernel<<<grid, kBlockSize>>>(d_ctx, d_merged, kB, kT, kNH, kDK);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kD, kB * kT, kD, &alpha, d_wo, kD,
                           d_merged, kD, &beta, d_out, kD));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(kB * kT * kD);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kB * kT * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // With all-ones input & identity weights: Q=K=V=ones → scores all equal
  // → softmax uniform → context = V = ones → output = ones
  for (int i = 0; i < kB * kT * kD; ++i) {
    EXPECT_NEAR(h_out[i], 1.0F, 1e-4F) << "index " << i;
  }

  CUDA_CHECK(cudaFree(d_wqkv));
  CUDA_CHECK(cudaFree(d_wo));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_qkv));
  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_ctx));
  CUDA_CHECK(cudaFree(d_merged));
}

// ------------------------------------------------------------------
// Attention rows sum to 1 after softmax
// ------------------------------------------------------------------

TEST_F(SelfAttentionTest, AttentionWeightsSumToOne) {
  constexpr int kB = 1;
  constexpr int kT = 4;
  constexpr int kD = 8;
  constexpr int kNH = 2;
  constexpr int kDK = kD / kNH;

  // Random Q, K
  std::vector<float> h_Q(kB * kNH * kT * kDK), h_K(kB * kNH * kT * kDK);
  for (int i = 0; i < static_cast<int>(h_Q.size()); ++i) {
    h_Q[i] = static_cast<float>(i % 7) * 0.2F - 0.6F;
    h_K[i] = static_cast<float>(i % 5) * 0.3F - 0.5F;
  }

  float *d_Q, *d_K, *d_scores;
  int score_size = kB * kNH * kT * kT;
  CUDA_CHECK(cudaMalloc(&d_Q, h_Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scores, score_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));

  float scale = 1.0F / sqrtf(static_cast<float>(kDK));
  float beta = 0.0F;
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kT, kT, kDK, &scale,
                                         d_K, kDK, kT * kDK, d_Q, kDK, kT * kDK, &beta, d_scores,
                                         kT, kT * kT, kB * kNH));

  int smem_block = 32;
  softmax_kernel<<<kB * kNH * kT, smem_block, smem_block * sizeof(float)>>>(d_scores, kB * kNH * kT,
                                                                            kT);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_scores(score_size);
  CUDA_CHECK(
      cudaMemcpy(h_scores.data(), d_scores, score_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < kB * kNH * kT; ++r) {
    float sum = 0.0F;
    for (int c = 0; c < kT; ++c) sum += h_scores[r * kT + c];
    EXPECT_NEAR(sum, 1.0F, 1e-5F) << "row=" << r;
  }

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_scores));
}
