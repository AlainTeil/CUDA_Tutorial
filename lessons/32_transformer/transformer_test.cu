/**
 * @file transformer_test.cu
 * @brief Unit tests for Lesson 32 — Transformer Encoder Block (Capstone).
 *
 * Tests verify the individual kernels (GELU, CLS pooling, cross-entropy
 * gradient, GELU backward, softmax backward) and the assembled forward
 * pass of the Transformer encoder.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
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

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  constexpr float kSqrt2OverPi = 0.7978845608F;
  float inner = kSqrt2OverPi * (x + 0.044715F * x * x * x);
  out[i] = 0.5F * x * (1.0F + tanhf(inner));
}

__global__ void cls_pool_kernel(const float* __restrict__ in, float* __restrict__ out, int B, int T,
                                int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * D) return;
  int b = idx / D;
  int d = idx % D;
  out[b * D + d] = in[b * T * D + d];
}

/// Negative-infinity fill used in softmax max-reduction.
static constexpr float kNegInf = -1e30F;

__global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float* rd = data + row * cols;
  float lmax = kNegInf;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) lmax = fmaxf(lmax, rd[c]);
  smem[threadIdx.x] = lmax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float mx = smem[0];
  float lsum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf(rd[c] - mx);
    rd[c] = e;
    lsum += e;
  }
  smem[threadIdx.x] = lsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float tot = smem[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) rd[c] /= tot;
}

__global__ void cross_entropy_grad_kernel(const float* __restrict__ probs,
                                          const int* __restrict__ labels,
                                          float* __restrict__ dlogits, int B, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * C) return;
  int b = idx / C;
  int c = idx % C;
  float indicator = (c == labels[b]) ? 1.0F : 0.0F;
  dlogits[idx] = (probs[idx] - indicator) / static_cast<float>(B);
}

__global__ void embedding_forward_kernel(const float* __restrict__ table,
                                         const int* __restrict__ ids, float* __restrict__ out,
                                         int total, int D, int V) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  int token_id = ids[row];
  assert(token_id >= 0 && token_id < V);
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
  data[idx] += (col % 2 == 0) ? sinf(angle) : cosf(angle);
}

__global__ void residual_add_kernel(const float* __restrict__ x, const float* __restrict__ residual,
                                    float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + residual[i];
}

__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ mean,
                                         float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;
  const float* ri = x + row * D;
  extern __shared__ float smem[];
  float ls = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) ls += ri[d];
  smem[threadIdx.x] = ls;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);
  float lv = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float diff = ri[d] - mu;
    lv += diff * diff;
  }
  smem[threadIdx.x] = lv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }
  float* ro = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float xh = (ri[d] - mu) * inv_std;
    ro[d] = gamma[d] * xh + beta[d];
  }
}

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

__global__ void merge_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK, int out_stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int in_idx = ((b * nH + h) * T + t) * dK + dk;
  int out_idx = b * (T * out_stride) + t * out_stride + h * dK + dk;
  out[out_idx] = in[in_idx];
}

// =============================================================================
// Tests
// =============================================================================

class TransformerTest : public ::testing::Test {
 protected:
  cublasHandle_t handle_{};
  void SetUp() override { CUBLAS_CHECK(cublasCreate(&handle_)); }
  void TearDown() override { CUBLAS_CHECK(cublasDestroy(handle_)); }
};

// ------------------------------------------------------------------
// GELU(0) = 0, GELU(large) ≈ x, GELU(-large) ≈ 0
// ------------------------------------------------------------------

TEST_F(TransformerTest, GELUBoundaryValues) {
  std::vector<float> h_in = {0.0F, 5.0F, -5.0F, 1.0F, -1.0F};
  int n = static_cast<int>(h_in.size());

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  gelu_kernel<<<1, kBlockSize>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(h_out[0], 0.0F, 1e-6F);  // GELU(0) = 0
  EXPECT_NEAR(h_out[1], 5.0F, 0.01F);  // GELU(large) ≈ x
  EXPECT_NEAR(h_out[2], 0.0F, 0.01F);  // GELU(-large) ≈ 0
  EXPECT_GT(h_out[3], 0.0F);           // GELU(1) > 0
  EXPECT_LT(h_out[4], 0.0F);           // GELU(-1) < 0

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

// ------------------------------------------------------------------
// CLS pooling extracts the first token
// ------------------------------------------------------------------

TEST_F(TransformerTest, CLSPoolExtractsFirstToken) {
  constexpr int kB = 2, kT = 4, kD = 8;
  int total = kB * kT * kD;

  std::vector<float> h_in(total);
  std::iota(h_in.begin(), h_in.end(), 0.0F);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, kB * kD * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (kB * kD + kBlockSize - 1) / kBlockSize;
  cls_pool_kernel<<<grid, kBlockSize>>>(d_in, d_out, kB, kT, kD);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(kB * kD);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kB * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // Batch 0, token 0: indices [0..D)
  for (int d = 0; d < kD; ++d) {
    EXPECT_FLOAT_EQ(h_out[d], static_cast<float>(d));
  }
  // Batch 1, token 0: starts at kT*kD
  for (int d = 0; d < kD; ++d) {
    EXPECT_FLOAT_EQ(h_out[kD + d], static_cast<float>(kT * kD + d));
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

// ------------------------------------------------------------------
// Cross‑entropy grad: correct class gets prob-1, others get prob
// ------------------------------------------------------------------

TEST_F(TransformerTest, CrossEntropyGradCorrectSign) {
  constexpr int kB = 2, kC = 3;
  // Uniform probs = 1/3
  std::vector<float> h_probs(kB * kC, 1.0F / 3.0F);
  std::vector<int> h_labels = {0, 2};

  float* d_probs;
  int* d_labels;
  float* d_dlogits;
  CUDA_CHECK(cudaMalloc(&d_probs, kB * kC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels, kB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dlogits, kB * kC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), kB * kC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), kB * sizeof(int), cudaMemcpyHostToDevice));

  int grid = (kB * kC + kBlockSize - 1) / kBlockSize;
  cross_entropy_grad_kernel<<<grid, kBlockSize>>>(d_probs, d_labels, d_dlogits, kB, kC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dlogits(kB * kC);
  CUDA_CHECK(
      cudaMemcpy(h_dlogits.data(), d_dlogits, kB * kC * sizeof(float), cudaMemcpyDeviceToHost));

  // For correct class: (1/3 - 1) / B = -2/(3B) < 0
  // For wrong class:   (1/3 - 0) / B  = 1/(3B) > 0
  EXPECT_LT(h_dlogits[0], 0.0F);  // batch 0, class 0 (correct)
  EXPECT_GT(h_dlogits[1], 0.0F);  // batch 0, class 1 (wrong)
  EXPECT_GT(h_dlogits[2], 0.0F);  // batch 0, class 2 (wrong)
  EXPECT_GT(h_dlogits[3], 0.0F);  // batch 1, class 0 (wrong)
  EXPECT_GT(h_dlogits[4], 0.0F);  // batch 1, class 1 (wrong)
  EXPECT_LT(h_dlogits[5], 0.0F);  // batch 1, class 2 (correct)

  CUDA_CHECK(cudaFree(d_probs));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_dlogits));
}

// ------------------------------------------------------------------
// Full forward pass produces finite logits
// ------------------------------------------------------------------

TEST_F(TransformerTest, ForwardProducesFiniteLogits) {
  constexpr int kV = 16;
  constexpr int kD = 16;
  constexpr int kT = 4;
  constexpr int kNH = 2;
  constexpr int kDK = kD / kNH;
  constexpr int kDFF = 32;
  constexpr int kNC = 2;
  constexpr int kB = 2;
  int BT = kB * kT;

  std::mt19937 rng(123);
  float fan = 1.0F / sqrtf(static_cast<float>(kD));

  auto alloc_init = [&](float** ptr, int sz, float sc) {
    CUDA_CHECK(cudaMalloc(ptr, sz * sizeof(float)));
    std::vector<float> h(sz);
    std::uniform_real_distribution<float> d(-sc, sc);
    for (auto& v : h) v = d(rng);
    CUDA_CHECK(cudaMemcpy(*ptr, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
  };
  auto alloc_const = [&](float** ptr, int sz, float val) {
    CUDA_CHECK(cudaMalloc(ptr, sz * sizeof(float)));
    std::vector<float> h(sz, val);
    CUDA_CHECK(cudaMemcpy(*ptr, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
  };

  // Weights
  float *emb, *wqkv, *wo, *lg, *lb, *w1, *b1, *w2, *b2, *lg2, *lb2, *wcls, *bcls;
  alloc_init(&emb, kV * kD, fan);
  alloc_init(&wqkv, 3 * kD * kD, fan);
  alloc_init(&wo, kD * kD, fan);
  alloc_const(&lg, kD, 1.0F);
  alloc_const(&lb, kD, 0.0F);
  alloc_init(&w1, kDFF * kD, fan);
  alloc_const(&b1, kDFF, 0.0F);
  alloc_init(&w2, kD * kDFF, fan);
  alloc_const(&b2, kD, 0.0F);
  alloc_const(&lg2, kD, 1.0F);
  alloc_const(&lb2, kD, 0.0F);
  alloc_init(&wcls, kNC * kD, fan);
  alloc_const(&bcls, kNC, 0.0F);

  // Scratch
  float *x_emb, *qkv_buf, *Q, *K, *V, *sc, *ctx, *ao, *r1, *l1o, *l1m, *l1r;
  float *fm, *fg, *fo, *r2, *l2o, *l2m, *l2r, *ci, *logits;
  CUDA_CHECK(cudaMalloc(&x_emb, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&qkv_buf, BT * 3 * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&Q, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sc, kB * kNH * kT * kT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ctx, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ao, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&r1, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l1o, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l1m, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l1r, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fm, BT * kDFF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fg, BT * kDFF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fo, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&r2, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l2o, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l2m, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&l2r, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ci, kB * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&logits, kB * kNC * sizeof(float)));

  // Input
  std::vector<int> h_ids(kB * kT);
  for (int i = 0; i < kB * kT; ++i) h_ids[i] = i % kV;
  int* d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, kB * kT * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), kB * kT * sizeof(int), cudaMemcpyHostToDevice));

  // Forward
  float alpha = 1.0F, beta_z = 0.0F;
  constexpr float kEps = 1e-5F;
  int BTD = BT * kD;

  int grid_emb = (BT * kD + kBlockSize - 1) / kBlockSize;
  embedding_forward_kernel<<<grid_emb, kBlockSize>>>(emb, d_ids, x_emb, BT, kD, kV);
  CUDA_CHECK(cudaGetLastError());
  sinusoidal_pe_kernel<<<grid_emb, kBlockSize>>>(x_emb, kT, kD, BT);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kD, BT, kD, &alpha, wqkv, kD,
                           x_emb, kD, &beta_z, qkv_buf, 3 * kD));

  int grid_h = (BTD + kBlockSize - 1) / kBlockSize;
  split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf, Q, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());
  split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf + kD, K, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());
  split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf + 2 * kD, V, kB, kT, kNH, kDK, 3 * kD);
  CUDA_CHECK(cudaGetLastError());

  float scale = 1.0F / sqrtf(static_cast<float>(kDK));
  CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kT, kT, kDK, &scale, K,
                                         kDK, kT * kDK, Q, kDK, kT * kDK, &beta_z, sc, kT, kT * kT,
                                         kB * kNH));

  softmax_kernel<<<kB * kNH * kT, 32, 32 * sizeof(float)>>>(sc, kB * kNH * kT, kT);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kDK, kT, kT, &alpha, V,
                                         kDK, kT * kDK, sc, kT, kT * kT, &beta_z, ctx, kDK,
                                         kT * kDK, kB * kNH));

  merge_heads_kernel<<<grid_h, kBlockSize>>>(ctx, ao, kB, kT, kNH, kDK, kD);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kD, BT, kD, &alpha, wo, kD, ao, kD,
                           &beta_z, ao, kD));

  residual_add_kernel<<<grid_h, kBlockSize>>>(ao, x_emb, r1, BTD);
  CUDA_CHECK(cudaGetLastError());
  layernorm_forward_kernel<<<BT, 32, 32 * sizeof(float)>>>(r1, l1o, lg, lb, l1m, l1r, BT, kD, kEps);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kDFF, BT, kD, &alpha, w1, kD, l1o, kD,
                           &beta_z, fm, kDFF));

  int gelu_grid = (BT * kDFF + kBlockSize - 1) / kBlockSize;
  gelu_kernel<<<gelu_grid, kBlockSize>>>(fm, fg, BT * kDFF);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kD, BT, kDFF, &alpha, w2, kDFF, fg,
                           kDFF, &beta_z, fo, kD));

  residual_add_kernel<<<grid_h, kBlockSize>>>(fo, l1o, r2, BTD);
  CUDA_CHECK(cudaGetLastError());
  layernorm_forward_kernel<<<BT, 32, 32 * sizeof(float)>>>(r2, l2o, lg2, lb2, l2m, l2r, BT, kD,
                                                           kEps);
  CUDA_CHECK(cudaGetLastError());

  int cls_grid = (kB * kD + kBlockSize - 1) / kBlockSize;
  cls_pool_kernel<<<cls_grid, kBlockSize>>>(l2o, ci, kB, kT, kD);
  CUDA_CHECK(cudaGetLastError());

  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, kNC, kB, kD, &alpha, wcls, kD, ci, kD,
                           &beta_z, logits, kNC));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check logits are finite
  std::vector<float> h_logits(kB * kNC);
  CUDA_CHECK(cudaMemcpy(h_logits.data(), logits, kB * kNC * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kB * kNC; ++i) {
    EXPECT_TRUE(std::isfinite(h_logits[i])) << "logit[" << i << "] is not finite";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(emb));
  CUDA_CHECK(cudaFree(wqkv));
  CUDA_CHECK(cudaFree(wo));
  CUDA_CHECK(cudaFree(lg));
  CUDA_CHECK(cudaFree(lb));
  CUDA_CHECK(cudaFree(w1));
  CUDA_CHECK(cudaFree(b1));
  CUDA_CHECK(cudaFree(w2));
  CUDA_CHECK(cudaFree(b2));
  CUDA_CHECK(cudaFree(lg2));
  CUDA_CHECK(cudaFree(lb2));
  CUDA_CHECK(cudaFree(wcls));
  CUDA_CHECK(cudaFree(bcls));
  CUDA_CHECK(cudaFree(x_emb));
  CUDA_CHECK(cudaFree(qkv_buf));
  CUDA_CHECK(cudaFree(Q));
  CUDA_CHECK(cudaFree(K));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(sc));
  CUDA_CHECK(cudaFree(ctx));
  CUDA_CHECK(cudaFree(ao));
  CUDA_CHECK(cudaFree(r1));
  CUDA_CHECK(cudaFree(l1o));
  CUDA_CHECK(cudaFree(l1m));
  CUDA_CHECK(cudaFree(l1r));
  CUDA_CHECK(cudaFree(fm));
  CUDA_CHECK(cudaFree(fg));
  CUDA_CHECK(cudaFree(fo));
  CUDA_CHECK(cudaFree(r2));
  CUDA_CHECK(cudaFree(l2o));
  CUDA_CHECK(cudaFree(l2m));
  CUDA_CHECK(cudaFree(l2r));
  CUDA_CHECK(cudaFree(ci));
  CUDA_CHECK(cudaFree(logits));
  CUDA_CHECK(cudaFree(d_ids));
}

// =============================================================================
// Backward kernels (duplicated — self-contained lesson)
// =============================================================================

__global__ void gelu_backward_kernel(const float* __restrict__ grad_out,
                                     const float* __restrict__ x_in, float* __restrict__ grad_in,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = x_in[i];
  constexpr float kA = 0.7978845608F;
  constexpr float kB = 0.044715F;
  float inner = kA * (x + kB * x * x * x);
  float th = tanhf(inner);
  float sech2 = 1.0F - th * th;
  float d_inner = kA * (1.0F + 3.0F * kB * x * x);
  float gelu_prime = 0.5F * (1.0F + th) + 0.5F * x * sech2 * d_inner;
  grad_in[i] = grad_out[i] * gelu_prime;
}

__global__ void softmax_backward_kernel(const float* __restrict__ grad_out,
                                        const float* __restrict__ softmax_out,
                                        float* __restrict__ grad_in, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  const float* dO = grad_out + row * cols;
  const float* S = softmax_out + row * cols;
  float* dI = grad_in + row * cols;
  float local_dot = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_dot += dO[c] * S[c];
  smem[threadIdx.x] = local_dot;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float dot = smem[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) dI[c] = S[c] * (dO[c] - dot);
}

// ------------------------------------------------------------------
// GELU backward: derivative matches finite-difference approximation
// ------------------------------------------------------------------

TEST_F(TransformerTest, GELUBackwardMatchesFiniteDiff) {
  std::vector<float> h_x = {0.0F, 1.0F, -1.0F, 2.0F, -0.5F};
  int n = static_cast<int>(h_x.size());

  float *d_x, *d_gelu, *d_grad_out, *d_grad_in;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gelu, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_in, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  // grad_out = 1 (pass-through to get derivative)
  std::vector<float> ones(n, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_grad_out, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  gelu_backward_kernel<<<1, kBlockSize>>>(d_grad_out, d_x, d_grad_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad(n);
  CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad_in, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Finite-difference approximation of gelu'(x)
  constexpr float kH = 1e-4F;
  for (int i = 0; i < n; ++i) {
    float x = h_x[i];
    auto gelu_cpu = [](float v) {
      float inner = 0.7978845608F * (v + 0.044715F * v * v * v);
      return 0.5F * v * (1.0F + tanhf(inner));
    };
    float fd = (gelu_cpu(x + kH) - gelu_cpu(x - kH)) / (2.0F * kH);
    EXPECT_NEAR(h_grad[i], fd, 1e-3F) << "GELU'(" << x << ") mismatch";
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_gelu));
  CUDA_CHECK(cudaFree(d_grad_out));
  CUDA_CHECK(cudaFree(d_grad_in));
}

// ------------------------------------------------------------------
// Softmax backward: Jacobian-vector product sums to zero per row
// ------------------------------------------------------------------

TEST_F(TransformerTest, SoftmaxBackwardSumsToZero) {
  constexpr int kRows = 4, kCols = 8;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  // Create softmax output (uniform = 1/kCols)
  std::vector<float> h_s(kRows * kCols, 1.0F / kCols);
  // Random upstream gradient
  std::vector<float> h_dO(kRows * kCols);
  for (auto& v : h_dO) v = dist(rng);

  float *d_s, *d_dO, *d_dI;
  CUDA_CHECK(cudaMalloc(&d_s, kRows * kCols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dO, kRows * kCols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dI, kRows * kCols * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), kRows * kCols * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dO, h_dO.data(), kRows * kCols * sizeof(float), cudaMemcpyHostToDevice));

  softmax_backward_kernel<<<kRows, 32, 32 * sizeof(float)>>>(d_dO, d_s, d_dI, kRows, kCols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dI(kRows * kCols);
  CUDA_CHECK(cudaMemcpy(h_dI.data(), d_dI, kRows * kCols * sizeof(float), cudaMemcpyDeviceToHost));

  // Property: sum of softmax backward output per row should be ~0
  // Because d/d_logit_j sum_j(softmax_j) = 0 (constraints of probability)
  for (int r = 0; r < kRows; ++r) {
    float row_sum = 0.0F;
    for (int c = 0; c < kCols; ++c) row_sum += h_dI[r * kCols + c];
    EXPECT_NEAR(row_sum, 0.0F, 1e-5F) << "Row " << r << " gradient sum != 0";
  }

  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_dO));
  CUDA_CHECK(cudaFree(d_dI));
}
