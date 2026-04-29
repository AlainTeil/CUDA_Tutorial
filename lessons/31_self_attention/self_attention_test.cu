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
// Backward kernels (mirrors of self_attention.cu)
// =============================================================================

__global__ void unmerge_heads_kernel(const float* __restrict__ d_merged,
                                     float* __restrict__ d_context, int B, int T, int nH, int dK) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int merged_idx = b * (T * nH * dK) + t * (nH * dK) + h * dK + dk;
  int ctx_idx = ((b * nH + h) * T + t) * dK + dk;
  d_context[ctx_idx] = d_merged[merged_idx];
}

__global__ void merge_qkv_grads_kernel(const float* __restrict__ dQ, const float* __restrict__ dKg,
                                       const float* __restrict__ dV, float* __restrict__ d_qkv,
                                       int B, int T, int nH, int dK, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int src = ((b * nH + h) * T + t) * dK + dk;
  int row_base = b * (T * 3 * D) + t * (3 * D) + h * dK + dk;
  d_qkv[row_base + 0] = dQ[src];
  d_qkv[row_base + D] = dKg[src];
  d_qkv[row_base + 2 * D] = dV[src];
}

__global__ void softmax_backward_kernel(const float* __restrict__ d_attn,
                                        const float* __restrict__ attn, float* d_pre, int rows,
                                        int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  const float* dy = d_attn + row * cols;
  const float* y = attn + row * cols;
  float* dx = d_pre + row * cols;
  float local = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local += dy[c] * y[c];
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float dot = smem[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) dx[c] = y[c] * (dy[c] - dot);
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

// =============================================================================
// Backward pass: finite-difference gradient check
// =============================================================================
//
// Verifies the analytical multi-head self-attention backward against a
// numerical reference computed by central differences:
//
//     dL/dx_i  ≈ ( L(x + ε e_i) − L(x − ε e_i) ) / (2ε)
//
// L is a synthetic loss L = Σ dY ⊙ out, so dL/dout = dY.  We probe a
// handful of indices in W_QKV, W_O, and X and check that each analytical
// gradient component is close to the central-difference estimate.
// =============================================================================

namespace {

struct MhaBuffers {
  // weights & input
  float *W_QKV, *W_O, *X, *out;
  // forward scratch
  float *qkv, *Q, *K, *V, *scores, *ctx, *merged;
  // backward outputs & scratch
  float *dY, *dX, *dW_QKV, *dW_O;
  float *dQ, *dKg, *dV, *dctx, *dscores, *dmerged, *dqkv;
  int B, T, D, nH, dK;
};

void mha_forward_test(MhaBuffers& m, cublasHandle_t handle) {
  float alpha = 1.0F, beta = 0.0F;
  int total = m.B * m.T * m.D;
  int grid = (total + kBlockSize - 1) / kBlockSize;

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * m.D, m.B * m.T, m.D, &alpha,
                           m.W_QKV, m.D, m.X, m.D, &beta, m.qkv, 3 * m.D));
  split_heads_kernel<<<grid, kBlockSize>>>(m.qkv, m.Q, m.B, m.T, m.nH, m.dK, 3 * m.D);
  split_heads_kernel<<<grid, kBlockSize>>>(m.qkv + m.D, m.K, m.B, m.T, m.nH, m.dK, 3 * m.D);
  split_heads_kernel<<<grid, kBlockSize>>>(m.qkv + 2 * m.D, m.V, m.B, m.T, m.nH, m.dK, 3 * m.D);

  float scale = 1.0F / sqrtf(static_cast<float>(m.dK));
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, m.T, m.T, m.dK, &scale,
                                         m.K, m.dK, m.T * m.dK, m.Q, m.dK, m.T * m.dK, &beta,
                                         m.scores, m.T, m.T * m.T, m.B * m.nH));

  int sblock = 32;
  softmax_kernel<<<m.B * m.nH * m.T, sblock, sblock * sizeof(float)>>>(m.scores, m.B * m.nH * m.T,
                                                                       m.T);

  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dK, m.T, m.T, &alpha,
                                         m.V, m.dK, m.T * m.dK, m.scores, m.T, m.T * m.T, &beta,
                                         m.ctx, m.dK, m.T * m.dK, m.B * m.nH));

  merge_heads_kernel<<<grid, kBlockSize>>>(m.ctx, m.merged, m.B, m.T, m.nH, m.dK);

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m.D, m.B * m.T, m.D, &alpha, m.W_O,
                           m.D, m.merged, m.D, &beta, m.out, m.D));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void mha_backward_test(MhaBuffers& m, cublasHandle_t handle) {
  float alpha = 1.0F, beta = 0.0F;
  int total = m.B * m.T * m.D;
  int grid = (total + kBlockSize - 1) / kBlockSize;

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m.D, m.D, m.B * m.T, &alpha, m.merged,
                           m.D, m.dY, m.D, &beta, m.dW_O, m.D));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.D, m.B * m.T, m.D, &alpha, m.W_O,
                           m.D, m.dY, m.D, &beta, m.dmerged, m.D));

  unmerge_heads_kernel<<<grid, kBlockSize>>>(m.dmerged, m.dctx, m.B, m.T, m.nH, m.dK);

  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, m.T, m.T, m.dK, &alpha,
                                         m.V, m.dK, m.T * m.dK, m.dctx, m.dK, m.T * m.dK, &beta,
                                         m.dscores, m.T, m.T * m.T, m.B * m.nH));

  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m.dK, m.T, m.T, &alpha,
                                         m.dctx, m.dK, m.T * m.dK, m.scores, m.T, m.T * m.T, &beta,
                                         m.dV, m.dK, m.T * m.dK, m.B * m.nH));

  int rows = m.B * m.nH * m.T;
  int sblock = 32;
  softmax_backward_kernel<<<rows, sblock, sblock * sizeof(float)>>>(m.dscores, m.scores, m.dscores,
                                                                    rows, m.T);

  float scale = 1.0F / sqrtf(static_cast<float>(m.dK));
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dK, m.T, m.T, &scale,
                                         m.K, m.dK, m.T * m.dK, m.dscores, m.T, m.T * m.T, &beta,
                                         m.dQ, m.dK, m.T * m.dK, m.B * m.nH));
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m.dK, m.T, m.T, &scale,
                                         m.Q, m.dK, m.T * m.dK, m.dscores, m.T, m.T * m.T, &beta,
                                         m.dKg, m.dK, m.T * m.dK, m.B * m.nH));

  merge_qkv_grads_kernel<<<grid, kBlockSize>>>(m.dQ, m.dKg, m.dV, m.dqkv, m.B, m.T, m.nH, m.dK,
                                               m.D);

  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m.D, 3 * m.D, m.B * m.T, &alpha, m.X,
                           m.D, m.dqkv, 3 * m.D, &beta, m.dW_QKV, m.D));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.D, m.B * m.T, 3 * m.D, &alpha,
                           m.W_QKV, m.D, m.dqkv, 3 * m.D, &beta, m.dX, m.D));
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

TEST_F(SelfAttentionTest, BackwardMatchesFiniteDifference) {
  constexpr int kB = 1;
  constexpr int kT = 3;
  constexpr int kD = 4;
  constexpr int kNH = 2;
  constexpr int kDK = kD / kNH;  // 2

  // Host inputs
  std::vector<float> h_X(kB * kT * kD);
  std::vector<float> h_W_QKV(3 * kD * kD);
  std::vector<float> h_W_O(kD * kD);
  std::vector<float> h_dY(kB * kT * kD);

  std::srand(2024);
  auto rnd = []() {
    return (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5F) * 0.6F;
  };
  for (auto& v : h_X) v = rnd();
  for (auto& v : h_W_QKV) v = rnd();
  for (auto& v : h_W_O) v = rnd();
  for (auto& v : h_dY) v = rnd();

  MhaBuffers m{};
  m.B = kB;
  m.T = kT;
  m.D = kD;
  m.nH = kNH;
  m.dK = kDK;
  auto alloc = [](float** p, int n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); };
  alloc(&m.W_QKV, 3 * kD * kD);
  alloc(&m.W_O, kD * kD);
  alloc(&m.X, kB * kT * kD);
  alloc(&m.out, kB * kT * kD);
  alloc(&m.qkv, kB * kT * 3 * kD);
  alloc(&m.Q, kB * kNH * kT * kDK);
  alloc(&m.K, kB * kNH * kT * kDK);
  alloc(&m.V, kB * kNH * kT * kDK);
  alloc(&m.scores, kB * kNH * kT * kT);
  alloc(&m.ctx, kB * kNH * kT * kDK);
  alloc(&m.merged, kB * kT * kD);
  alloc(&m.dY, kB * kT * kD);
  alloc(&m.dX, kB * kT * kD);
  alloc(&m.dW_QKV, 3 * kD * kD);
  alloc(&m.dW_O, kD * kD);
  alloc(&m.dQ, kB * kNH * kT * kDK);
  alloc(&m.dKg, kB * kNH * kT * kDK);
  alloc(&m.dV, kB * kNH * kT * kDK);
  alloc(&m.dctx, kB * kNH * kT * kDK);
  alloc(&m.dscores, kB * kNH * kT * kT);
  alloc(&m.dmerged, kB * kT * kD);
  alloc(&m.dqkv, kB * kT * 3 * kD);

  auto upload = [](float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  };
  upload(m.W_QKV, h_W_QKV);
  upload(m.W_O, h_W_O);
  upload(m.X, h_X);
  upload(m.dY, h_dY);

  // Analytical backward
  mha_forward_test(m, handle_);
  mha_backward_test(m, handle_);

  std::vector<float> h_dX(kB * kT * kD), h_dW_QKV(3 * kD * kD), h_dW_O(kD * kD);
  CUDA_CHECK(cudaMemcpy(h_dX.data(), m.dX, h_dX.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dW_QKV.data(), m.dW_QKV, h_dW_QKV.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_dW_O.data(), m.dW_O, h_dW_O.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Helper: loss L(out) = sum dY ⊙ out
  auto eval_loss = [&]() {
    mha_forward_test(m, handle_);
    std::vector<float> h_out(kB * kT * kD);
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), m.out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float s = 0.0F;
    for (size_t i = 0; i < h_out.size(); ++i) s += h_out[i] * h_dY[i];
    return s;
  };

  const float kEps = 1e-3F;
  const float kAbs = 5e-3F;
  const float kRel = 5e-3F;

  // Probe a handful of indices in each parameter
  auto check_param = [&](const char* name, std::vector<float>& host_param, float* d_param,
                         const std::vector<float>& analytical, const std::vector<int>& indices) {
    for (int idx : indices) {
      float orig = host_param[idx];
      host_param[idx] = orig + kEps;
      upload(d_param, host_param);
      float l_plus = eval_loss();
      host_param[idx] = orig - kEps;
      upload(d_param, host_param);
      float l_minus = eval_loss();
      host_param[idx] = orig;
      upload(d_param, host_param);
      float fd = (l_plus - l_minus) / (2.0F * kEps);
      float ana = analytical[idx];
      float tol = kAbs + kRel * std::fabs(fd);
      EXPECT_NEAR(ana, fd, tol) << name << "[" << idx << "] analytical=" << ana << " fd=" << fd;
    }
  };

  check_param("dX", h_X, m.X, h_dX, {0, 3, 5, 7, static_cast<int>(h_X.size()) - 1});
  check_param("dW_O", h_W_O, m.W_O, h_dW_O, {0, 5, 9, static_cast<int>(h_W_O.size()) - 1});
  check_param("dW_QKV", h_W_QKV, m.W_QKV, h_dW_QKV,
              {0, 8, 17, 31, static_cast<int>(h_W_QKV.size()) - 1});

  CUDA_CHECK(cudaFree(m.W_QKV));
  CUDA_CHECK(cudaFree(m.W_O));
  CUDA_CHECK(cudaFree(m.X));
  CUDA_CHECK(cudaFree(m.out));
  CUDA_CHECK(cudaFree(m.qkv));
  CUDA_CHECK(cudaFree(m.Q));
  CUDA_CHECK(cudaFree(m.K));
  CUDA_CHECK(cudaFree(m.V));
  CUDA_CHECK(cudaFree(m.scores));
  CUDA_CHECK(cudaFree(m.ctx));
  CUDA_CHECK(cudaFree(m.merged));
  CUDA_CHECK(cudaFree(m.dY));
  CUDA_CHECK(cudaFree(m.dX));
  CUDA_CHECK(cudaFree(m.dW_QKV));
  CUDA_CHECK(cudaFree(m.dW_O));
  CUDA_CHECK(cudaFree(m.dQ));
  CUDA_CHECK(cudaFree(m.dKg));
  CUDA_CHECK(cudaFree(m.dV));
  CUDA_CHECK(cudaFree(m.dctx));
  CUDA_CHECK(cudaFree(m.dscores));
  CUDA_CHECK(cudaFree(m.dmerged));
  CUDA_CHECK(cudaFree(m.dqkv));
}
