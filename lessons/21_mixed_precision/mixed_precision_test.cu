/**
 * @file mixed_precision_test.cu
 * @brief Unit tests for Lesson 21 — Mixed-Precision Training.
 *
 * Tests cover:
 *   - FP16 ↔ FP32 round-trip conversion accuracy
 *   - FP16 GEMM correctness (cublasGemmEx vs CPU reference)
 *   - TF32 GEMM correctness (cublasSgemm with TF32 math mode)
 *   - Loss-scaling round-trip (scale → compute → unscale)
 *   - FP32 training baseline (loss decreases)
 *   - TF32 training (loss decreases, accuracy ≥ 80%)
 *   - FP16 training (loss decreases, accuracy ≥ 80%)
 *   - All three modes reach high accuracy (≥ 90%)
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// =============================================================================
// Error-checking macros (test versions — use ASSERT for cleaner output)
// =============================================================================

#define CUDA_ASSERT(call) ASSERT_EQ((call), cudaSuccess)
#define CUBLAS_ASSERT(call) ASSERT_EQ((call), CUBLAS_STATUS_SUCCESS)

// Abort-style macros (needed by functions shared with the main source).
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                          \
  do {                                                                              \
    cublasStatus_t st_ = (call);                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                             \
      std::fprintf(stderr, "cuBLAS error at %s:%d — code %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                          \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

// =============================================================================
// Precision mode (must match main source)
// =============================================================================

enum class PrecisionMode { kFP32, kTF32, kFP16 };

// =============================================================================
// Conversion kernels (duplicated from main source for standalone compilation)
// =============================================================================

__global__ void fp32_to_fp16_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void fp16_to_fp32_kernel(const __half* __restrict__ src, float* __restrict__ dst,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}

static void fp32_to_fp16(const float* d_src, __half* d_dst, int n) {
  int blk = 256;
  fp32_to_fp16_kernel<<<(n + blk - 1) / blk, blk>>>(d_src, d_dst, n);
}

static void fp16_to_fp32(const __half* d_src, float* d_dst, int n) {
  int blk = 256;
  fp16_to_fp32_kernel<<<(n + blk - 1) / blk, blk>>>(d_src, d_dst, n);
}

// =============================================================================
// GEMM wrappers (duplicated for standalone compilation)
// =============================================================================

static void gemm_fp32(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                      const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

static void gemm_fp16(cublasHandle_t h, int M, int K, int N, float alpha, const __half* A,
                      const __half* B, float beta, __half* C) {
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
                            CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void gemm_fp16_AT(cublasHandle_t h, int M, int K, int N, float alpha, const __half* A,
                         const __half* B, float beta, __half* C) {
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, CUDA_R_16F, N, A,
                            CUDA_R_16F, M, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void gemm_fp32_AT(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

// =============================================================================
// Element-wise kernels (duplicated for standalone compilation)
// =============================================================================

__global__ void relu_fwd(const float* __restrict__ x, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(x[i], 0.0F);
}

__global__ void relu_bwd(const float* __restrict__ z, const float* __restrict__ dy,
                         float* __restrict__ dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = (z[i] > 0.0F) ? dy[i] : 0.0F;
}

__global__ void add_bias(float* __restrict__ out, const float* __restrict__ bias, int B, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * D) out[idx] += bias[idx % D];
}

__global__ void ce_bwd(const float* __restrict__ log_sm, const float* __restrict__ target,
                       float* __restrict__ dz, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dz[i] = expf(log_sm[i]) - target[i];
}

__global__ void col_sum(const float* __restrict__ in, float* __restrict__ out, int B, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) {
    float s = 0.0F;
    for (int b = 0; b < B; ++b) s += in[b * D + j];
    out[j] = s;
  }
}

__global__ void scale_kernel(float* __restrict__ data, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= s;
}

__global__ void batch_log_softmax(const float* __restrict__ logits, float* __restrict__ log_sm,
                                  int C) {
  int b = blockIdx.x;
  const float* row = logits + b * C;
  float* out = log_sm + b * C;
  float mx = row[0];
  for (int c = 1; c < C; ++c) mx = fmaxf(mx, row[c]);
  float sum = 0.0F;
  for (int c = 0; c < C; ++c) sum += expf(row[c] - mx);
  float log_sum = logf(sum) + mx;
  for (int c = 0; c < C; ++c) out[c] = row[c] - log_sum;
}

__global__ void ce_loss_kernel(const float* __restrict__ log_sm, const float* __restrict__ target,
                               float* __restrict__ loss, int B, int C) {
  float s = 0.0F;
  for (int i = 0; i < B * C; ++i) s -= target[i] * log_sm[i];
  *loss = s / static_cast<float>(B);
}

__global__ void sgd_update(float* __restrict__ W, const float* __restrict__ dW, float lr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) W[i] -= lr * dW[i];
}

// =============================================================================
// MixedPrecisionMLP (duplicated for standalone test compilation)
// =============================================================================

struct MixedPrecisionMLP {
  int in_dim{}, hid_dim{}, out_dim{}, max_batch{};
  PrecisionMode mode{PrecisionMode::kFP32};
  float loss_scale{1.0F};
  cublasHandle_t cublas{};
  float *W1{}, *b1{}, *W2{}, *b2{};
  float *Z1{}, *A1{}, *Z2{}, *LS{};
  float *dZ2{}, *dA1{}, *dZ1{};
  float *dW1{}, *db1{}, *dW2{}, *db2{};
  __half *W1_h{}, *W2_h{}, *X_h{};
  __half *Z1_h{}, *A1_h{}, *Z2_h{};
  __half *dZ2_h{}, *dA1_h{}, *dZ1_h{};
  __half *dW2_h{}, *dW1_h{};

  void alloc(int in, int hid, int out, int max_B, PrecisionMode p, float ls = 1024.0F) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    max_batch = max_B;
    mode = p;
    loss_scale = (p == PrecisionMode::kFP16) ? ls : 1.0F;
    CUBLAS_CHECK(cublasCreate(&cublas));
    if (mode == PrecisionMode::kTF32) {
      CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    } else {
      CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
    }
    auto alloc_f = [](float** ptr, size_t count) {
      CUDA_CHECK(cudaMalloc(ptr, count * sizeof(float)));
    };
    alloc_f(&W1, static_cast<size_t>(in) * hid);
    alloc_f(&b1, hid);
    alloc_f(&W2, static_cast<size_t>(hid) * out);
    alloc_f(&b2, out);
    alloc_f(&Z1, static_cast<size_t>(max_B) * hid);
    alloc_f(&A1, static_cast<size_t>(max_B) * hid);
    alloc_f(&Z2, static_cast<size_t>(max_B) * out);
    alloc_f(&LS, static_cast<size_t>(max_B) * out);
    alloc_f(&dZ2, static_cast<size_t>(max_B) * out);
    alloc_f(&dA1, static_cast<size_t>(max_B) * hid);
    alloc_f(&dZ1, static_cast<size_t>(max_B) * hid);
    alloc_f(&dW1, static_cast<size_t>(in) * hid);
    alloc_f(&db1, hid);
    alloc_f(&dW2, static_cast<size_t>(hid) * out);
    alloc_f(&db2, out);
    if (mode == PrecisionMode::kFP16) {
      auto alloc_h = [](__half** ptr, size_t count) {
        CUDA_CHECK(cudaMalloc(ptr, count * sizeof(__half)));
      };
      alloc_h(&W1_h, static_cast<size_t>(in) * hid);
      alloc_h(&W2_h, static_cast<size_t>(hid) * out);
      alloc_h(&X_h, static_cast<size_t>(max_B) * in);
      alloc_h(&Z1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&A1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&Z2_h, static_cast<size_t>(max_B) * out);
      alloc_h(&dZ2_h, static_cast<size_t>(max_B) * out);
      alloc_h(&dA1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&dZ1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&dW2_h, static_cast<size_t>(hid) * out);
      alloc_h(&dW1_h, static_cast<size_t>(in) * hid);
    }
  }

  void init_weights(unsigned seed) {
    std::mt19937 rng(seed);
    auto xavier = [&](float* d_buf, int fan_in, int fan_out) {
      int n = fan_in * fan_out;
      float limit = std::sqrt(6.0F / static_cast<float>(fan_in + fan_out));
      std::uniform_real_distribution<float> dist(-limit, limit);
      std::vector<float> host(static_cast<size_t>(n));
      for (auto& v : host) v = dist(rng);
      CUDA_CHECK(cudaMemcpy(d_buf, host.data(), static_cast<size_t>(n) * sizeof(float),
                            cudaMemcpyHostToDevice));
    };
    xavier(W1, in_dim, hid_dim);
    xavier(W2, hid_dim, out_dim);
    CUDA_CHECK(cudaMemset(b1, 0, static_cast<size_t>(hid_dim) * sizeof(float)));
    CUDA_CHECK(cudaMemset(b2, 0, static_cast<size_t>(out_dim) * sizeof(float)));
  }

  float forward_fp32(const float* d_X, const float* d_target, int B) {
    int blk = 256;
    gemm_fp32(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);
    gemm_fp32(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    batch_log_softmax<<<B, 1>>>(Z2, LS, out_dim);
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    ce_loss_kernel<<<1, 1>>>(LS, d_target, d_loss, B, out_dim);
    float h_loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss;
  }

  float forward_fp16(const float* d_X, const float* d_target, int B) {
    int blk = 256;
    fp32_to_fp16(W1, W1_h, in_dim * hid_dim);
    fp32_to_fp16(W2, W2_h, hid_dim * out_dim);
    fp32_to_fp16(d_X, X_h, B * in_dim);
    gemm_fp16(cublas, B, in_dim, hid_dim, 1.0F, X_h, W1_h, 0.0F, Z1_h);
    fp16_to_fp32(Z1_h, Z1, B * hid_dim);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);
    fp32_to_fp16(A1, A1_h, B * hid_dim);
    gemm_fp16(cublas, B, hid_dim, out_dim, 1.0F, A1_h, W2_h, 0.0F, Z2_h);
    fp16_to_fp32(Z2_h, Z2, B * out_dim);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    batch_log_softmax<<<B, 1>>>(Z2, LS, out_dim);
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    ce_loss_kernel<<<1, 1>>>(LS, d_target, d_loss, B, out_dim);
    float h_loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss;
  }

  float forward(const float* d_X, const float* d_target, int B) {
    if (mode == PrecisionMode::kFP16) return forward_fp16(d_X, d_target, B);
    return forward_fp32(d_X, d_target, B);
  }

  void backward_fp32(const float* d_X, const float* d_target, int B) {
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);
    ce_bwd<<<(B * out_dim + blk - 1) / blk, blk>>>(LS, d_target, dZ2, B * out_dim);
    gemm_fp32_AT(cublas, hid_dim, B, out_dim, inv_B, A1, dZ2, 0.0F, dW2);
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);
    {
      float one = 1.0F, zero = 0.0F;
      CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2,
                               out_dim, dZ2, out_dim, &zero, dA1, hid_dim));
    }
    relu_bwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, B * hid_dim);
    gemm_fp32_AT(cublas, in_dim, B, hid_dim, inv_B, d_X, dZ1, 0.0F, dW1);
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward_fp16(const float* d_X, const float* d_target, int B) {
    (void)d_X;  // FP16 path uses X_h (converted during forward).
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);
    ce_bwd<<<(B * out_dim + blk - 1) / blk, blk>>>(LS, d_target, dZ2, B * out_dim);
    scale_kernel<<<(B * out_dim + blk - 1) / blk, blk>>>(dZ2, loss_scale, B * out_dim);
    fp32_to_fp16(dZ2, dZ2_h, B * out_dim);
    gemm_fp16_AT(cublas, hid_dim, B, out_dim, inv_B, A1_h, dZ2_h, 0.0F, dW2_h);
    fp16_to_fp32(dW2_h, dW2, hid_dim * out_dim);
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);
    {
      float one = 1.0F, zero = 0.0F;
      CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2_h,
                                CUDA_R_16F, out_dim, dZ2_h, CUDA_R_16F, out_dim, &zero, dA1_h,
                                CUDA_R_16F, hid_dim, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    fp16_to_fp32(dA1_h, dA1, B * hid_dim);
    relu_bwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, B * hid_dim);
    fp32_to_fp16(dZ1, dZ1_h, B * hid_dim);
    gemm_fp16_AT(cublas, in_dim, B, hid_dim, inv_B, X_h, dZ1_h, 0.0F, dW1_h);
    fp16_to_fp32(dW1_h, dW1, in_dim * hid_dim);
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);
    float inv_scale = 1.0F / loss_scale;
    int n_dW2 = hid_dim * out_dim;
    int n_dW1 = in_dim * hid_dim;
    scale_kernel<<<(n_dW2 + blk - 1) / blk, blk>>>(dW2, inv_scale, n_dW2);
    scale_kernel<<<(n_dW1 + blk - 1) / blk, blk>>>(dW1, inv_scale, n_dW1);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_scale, out_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_scale, hid_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(const float* d_X, const float* d_target, int B) {
    if (mode == PrecisionMode::kFP16)
      backward_fp16(d_X, d_target, B);
    else
      backward_fp32(d_X, d_target, B);
  }

  void sgd_step(float lr) {
    int blk = 256;
    int n1 = in_dim * hid_dim, n2 = hid_dim * out_dim;
    sgd_update<<<(n1 + blk - 1) / blk, blk>>>(W1, dW1, lr, n1);
    sgd_update<<<(hid_dim + blk - 1) / blk, blk>>>(b1, db1, lr, hid_dim);
    sgd_update<<<(n2 + blk - 1) / blk, blk>>>(W2, dW2, lr, n2);
    sgd_update<<<(out_dim + blk - 1) / blk, blk>>>(b2, db2, lr, out_dim);
  }

  std::vector<int> predict_batch(const float* d_X, int B) {
    int blk = 256;
    gemm_fp32(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);
    gemm_fp32(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> logits(static_cast<size_t>(B) * out_dim);
    CUDA_CHECK(cudaMemcpy(logits.data(), Z2, static_cast<size_t>(B) * out_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<int> preds(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
      int best = 0;
      float best_val = logits[static_cast<size_t>(b) * out_dim];
      for (int c = 1; c < out_dim; ++c) {
        float v = logits[static_cast<size_t>(b) * out_dim + c];
        if (v > best_val) {
          best_val = v;
          best = c;
        }
      }
      preds[static_cast<size_t>(b)] = best;
    }
    return preds;
  }

  void free_all() {
    auto safe_free = [](void* p) {
      if (p) CUDA_CHECK(cudaFree(p));
    };
    safe_free(W1);
    safe_free(b1);
    safe_free(W2);
    safe_free(b2);
    safe_free(Z1);
    safe_free(A1);
    safe_free(Z2);
    safe_free(LS);
    safe_free(dZ2);
    safe_free(dA1);
    safe_free(dZ1);
    safe_free(dW1);
    safe_free(db1);
    safe_free(dW2);
    safe_free(db2);
    safe_free(W1_h);
    safe_free(W2_h);
    safe_free(X_h);
    safe_free(Z1_h);
    safe_free(A1_h);
    safe_free(Z2_h);
    safe_free(dZ2_h);
    safe_free(dA1_h);
    safe_free(dZ1_h);
    safe_free(dW2_h);
    safe_free(dW1_h);
    if (cublas) CUBLAS_CHECK(cublasDestroy(cublas));
    cublas = nullptr;
  }
};

// =============================================================================
// Data generation (same as Lessons 17, 20, 21)
// =============================================================================

static void generate_data(std::vector<float>& flat, std::vector<int>& labels, int n_per_class,
                          unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  const float centres[3][4] = {
      {0.0F, 0.0F, 0.0F, 0.0F},
      {3.0F, 3.0F, 3.0F, 3.0F},
      {-3.0F, 3.0F, -3.0F, 3.0F},
  };
  flat.clear();
  labels.clear();
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      for (int f = 0; f < 4; ++f) flat.push_back(centres[c][f] + noise(rng));
      labels.push_back(c);
    }
  }
}

static std::vector<float> make_one_hot(const std::vector<int>& labels, int C) {
  size_t N = labels.size();
  std::vector<float> oh(N * static_cast<size_t>(C), 0.0F);
  for (size_t i = 0; i < N; ++i)
    oh[i * static_cast<size_t>(C) + static_cast<size_t>(labels[i])] = 1.0F;
  return oh;
}

// =============================================================================
// Training helper used by multiple tests
// =============================================================================

struct TrainResult {
  float first_loss;
  float last_loss;
  float accuracy;
};

static TrainResult run_training(PrecisionMode mode, int epochs = 50, float lr = 0.1F,
                                int n_per_class = 50, int batch_size = 32) {
  int N_TOTAL = 3 * n_per_class;

  std::vector<float> flat_data;
  std::vector<int> labels;
  generate_data(flat_data, labels, n_per_class, 42);
  std::vector<float> one_hot = make_one_hot(labels, 3);

  MixedPrecisionMLP mlp{};
  mlp.alloc(4, 16, 3, batch_size, mode);
  mlp.init_weights(123);

  float *d_bx, *d_bt;
  CUDA_CHECK(cudaMalloc(&d_bx, static_cast<size_t>(batch_size) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bt, static_cast<size_t>(batch_size) * 3 * sizeof(float)));

  std::vector<int> indices(static_cast<size_t>(N_TOTAL));
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(7);

  float first_epoch_loss = 0.0F;
  float last_epoch_loss = 0.0F;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);
    float epoch_loss = 0.0F;
    int n_batches = 0;
    for (int start = 0; start + batch_size <= N_TOTAL; start += batch_size) {
      std::vector<float> bx(static_cast<size_t>(batch_size) * 4);
      std::vector<float> bt(static_cast<size_t>(batch_size) * 3);
      for (int b = 0; b < batch_size; ++b) {
        int si = indices[static_cast<size_t>(start + b)];
        std::memcpy(&bx[static_cast<size_t>(b) * 4], &flat_data[static_cast<size_t>(si) * 4],
                    4 * sizeof(float));
        std::memcpy(&bt[static_cast<size_t>(b) * 3], &one_hot[static_cast<size_t>(si) * 3],
                    3 * sizeof(float));
      }
      CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), static_cast<size_t>(batch_size) * 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), static_cast<size_t>(batch_size) * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));
      float loss = mlp.forward(d_bx, d_bt, batch_size);
      mlp.backward(d_bx, d_bt, batch_size);
      mlp.sgd_step(lr);
      epoch_loss += loss;
      ++n_batches;
    }
    float avg = epoch_loss / static_cast<float>(n_batches);
    if (epoch == 0) first_epoch_loss = avg;
    last_epoch_loss = avg;
  }

  // Evaluate accuracy.
  float* d_all;
  CUDA_CHECK(cudaMalloc(&d_all, static_cast<size_t>(N_TOTAL) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_all, flat_data.data(), static_cast<size_t>(N_TOTAL) * 4 * sizeof(float),
                        cudaMemcpyHostToDevice));
  int correct = 0;
  int evaluated = 0;
  for (int start = 0; start + batch_size <= N_TOTAL; start += batch_size) {
    auto preds = mlp.predict_batch(d_all + start * 4, batch_size);
    for (int b = 0; b < batch_size; ++b) {
      if (preds[static_cast<size_t>(b)] == labels[static_cast<size_t>(start + b)]) ++correct;
    }
    evaluated += batch_size;
  }
  float accuracy = static_cast<float>(correct) / static_cast<float>(evaluated) * 100.0F;

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_bx));
  CUDA_CHECK(cudaFree(d_bt));
  CUDA_CHECK(cudaFree(d_all));

  return {first_epoch_loss, last_epoch_loss, accuracy};
}

// =============================================================================
// Tests
// =============================================================================

/// Test FP32 → FP16 → FP32 round-trip preserves values within FP16 precision.
TEST(MixedPrecisionTest, FP16RoundTrip) {
  constexpr int N = 1024;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0F, 10.0F);

  std::vector<float> host_in(N);
  for (auto& v : host_in) v = dist(rng);

  float* d_fp32_in;
  __half* d_fp16;
  float* d_fp32_out;
  CUDA_ASSERT(cudaMalloc(&d_fp32_in, N * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_fp16, N * sizeof(__half)));
  CUDA_ASSERT(cudaMalloc(&d_fp32_out, N * sizeof(float)));

  CUDA_ASSERT(cudaMemcpy(d_fp32_in, host_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  fp32_to_fp16(d_fp32_in, d_fp16, N);
  fp16_to_fp32(d_fp16, d_fp32_out, N);
  CUDA_ASSERT(cudaDeviceSynchronize());

  std::vector<float> host_out(N);
  CUDA_ASSERT(cudaMemcpy(host_out.data(), d_fp32_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  // FP16 has ~0.1% relative error for values in [-10, 10].
  for (int i = 0; i < N; ++i) {
    float expected = host_in[static_cast<size_t>(i)];
    float actual = host_out[static_cast<size_t>(i)];
    EXPECT_NEAR(actual, expected, std::fabs(expected) * 0.002F + 1e-4F)
        << "Mismatch at index " << i;
  }

  CUDA_ASSERT(cudaFree(d_fp32_in));
  CUDA_ASSERT(cudaFree(d_fp16));
  CUDA_ASSERT(cudaFree(d_fp32_out));
}

/// Test FP16 GEMM (cublasGemmEx) matches CPU reference within FP16 tolerance.
TEST(MixedPrecisionTest, FP16GemmMatchesCPU) {
  constexpr int M = 8, K = 4, N = 3;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  // Generate random FP32 matrices.
  std::vector<float> h_A(M * K), h_B(K * N);
  for (auto& v : h_A) v = dist(rng);
  for (auto& v : h_B) v = dist(rng);

  // CPU reference: C = A · B (row-major).
  std::vector<float> h_C_ref(M * N, 0.0F);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        h_C_ref[static_cast<size_t>(i) * N + j] +=
            h_A[static_cast<size_t>(i) * K + k] * h_B[static_cast<size_t>(k) * N + j];

  // GPU: convert to FP16, run GemmEx, convert back.
  float *d_A32, *d_B32, *d_C32;
  __half *d_A16, *d_B16, *d_C16;
  CUDA_ASSERT(cudaMalloc(&d_A32, M * K * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_B32, K * N * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_C32, M * N * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_A16, M * K * sizeof(__half)));
  CUDA_ASSERT(cudaMalloc(&d_B16, K * N * sizeof(__half)));
  CUDA_ASSERT(cudaMalloc(&d_C16, M * N * sizeof(__half)));

  CUDA_ASSERT(cudaMemcpy(d_A32, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_ASSERT(cudaMemcpy(d_B32, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  fp32_to_fp16(d_A32, d_A16, M * K);
  fp32_to_fp16(d_B32, d_B16, K * N);

  cublasHandle_t handle;
  CUBLAS_ASSERT(cublasCreate(&handle));

  gemm_fp16(handle, M, K, N, 1.0F, d_A16, d_B16, 0.0F, d_C16);

  fp16_to_fp32(d_C16, d_C32, M * N);
  CUDA_ASSERT(cudaDeviceSynchronize());

  std::vector<float> h_C_gpu(M * N);
  CUDA_ASSERT(cudaMemcpy(h_C_gpu.data(), d_C32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // FP16 GEMM should match CPU within ~1% relative error.
  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(h_C_gpu[static_cast<size_t>(i)], h_C_ref[static_cast<size_t>(i)],
                std::fabs(h_C_ref[static_cast<size_t>(i)]) * 0.02F + 0.01F)
        << "Mismatch at index " << i;
  }

  CUBLAS_ASSERT(cublasDestroy(handle));
  CUDA_ASSERT(cudaFree(d_A32));
  CUDA_ASSERT(cudaFree(d_B32));
  CUDA_ASSERT(cudaFree(d_C32));
  CUDA_ASSERT(cudaFree(d_A16));
  CUDA_ASSERT(cudaFree(d_B16));
  CUDA_ASSERT(cudaFree(d_C16));
}

/// Test TF32 GEMM matches FP32 GEMM within TF32 tolerance.
TEST(MixedPrecisionTest, TF32GemmMatchesFP32) {
  constexpr int M = 8, K = 4, N = 3;
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  std::vector<float> h_A(M * K), h_B(K * N);
  for (auto& v : h_A) v = dist(rng);
  for (auto& v : h_B) v = dist(rng);

  // CPU reference.
  std::vector<float> h_C_ref(M * N, 0.0F);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        h_C_ref[static_cast<size_t>(i) * N + j] +=
            h_A[static_cast<size_t>(i) * K + k] * h_B[static_cast<size_t>(k) * N + j];

  float *d_A, *d_B, *d_C;
  CUDA_ASSERT(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_ASSERT(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_ASSERT(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_ASSERT(cublasCreate(&handle));
  CUBLAS_ASSERT(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  gemm_fp32(handle, M, K, N, 1.0F, d_A, d_B, 0.0F, d_C);
  CUDA_ASSERT(cudaDeviceSynchronize());

  std::vector<float> h_C_gpu(M * N);
  CUDA_ASSERT(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // TF32 truncates mantissa from 23 to 10 bits → ~0.1% relative error.
  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(h_C_gpu[static_cast<size_t>(i)], h_C_ref[static_cast<size_t>(i)],
                std::fabs(h_C_ref[static_cast<size_t>(i)]) * 0.01F + 0.001F)
        << "Mismatch at index " << i;
  }

  CUBLAS_ASSERT(cublasDestroy(handle));
  CUDA_ASSERT(cudaFree(d_A));
  CUDA_ASSERT(cudaFree(d_B));
  CUDA_ASSERT(cudaFree(d_C));
}

/// Test loss-scaling round-trip: scale up → compute → scale down = original.
TEST(MixedPrecisionTest, LossScalingRoundTrip) {
  constexpr int N = 256;
  constexpr float SCALE = 1024.0F;
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-0.01F, 0.01F);

  // Small gradient values typical of deep networks.
  std::vector<float> h_grads(N);
  for (auto& v : h_grads) v = dist(rng);

  float* d_grads;
  CUDA_ASSERT(cudaMalloc(&d_grads, N * sizeof(float)));
  CUDA_ASSERT(cudaMemcpy(d_grads, h_grads.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Scale up (simulating loss scaling).
  int blk = 256;
  scale_kernel<<<(N + blk - 1) / blk, blk>>>(d_grads, SCALE, N);

  // Scale down (un-scaling).
  scale_kernel<<<(N + blk - 1) / blk, blk>>>(d_grads, 1.0F / SCALE, N);
  CUDA_ASSERT(cudaDeviceSynchronize());

  std::vector<float> h_out(N);
  CUDA_ASSERT(cudaMemcpy(h_out.data(), d_grads, N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(h_out[static_cast<size_t>(i)], h_grads[static_cast<size_t>(i)], 1e-6F)
        << "Loss scaling round-trip failed at index " << i;
  }

  CUDA_ASSERT(cudaFree(d_grads));
}

/// Test FP32 baseline training: loss should decrease.
TEST(MixedPrecisionTest, FP32LossDecreases) {
  auto result = run_training(PrecisionMode::kFP32, 30);
  EXPECT_LT(result.last_loss, result.first_loss * 0.5F)
      << "FP32 loss should decrease significantly";
}

/// Test TF32 training: loss should decrease.
TEST(MixedPrecisionTest, TF32LossDecreases) {
  auto result = run_training(PrecisionMode::kTF32, 30);
  EXPECT_LT(result.last_loss, result.first_loss * 0.5F)
      << "TF32 loss should decrease significantly";
}

/// Test FP16 training: loss should decrease.
TEST(MixedPrecisionTest, FP16LossDecreases) {
  auto result = run_training(PrecisionMode::kFP16, 30);
  EXPECT_LT(result.last_loss, result.first_loss * 0.5F)
      << "FP16 loss should decrease significantly";
}

/// Test FP32 reaches high accuracy.
TEST(MixedPrecisionTest, FP32HighAccuracy) {
  auto result = run_training(PrecisionMode::kFP32, 50);
  EXPECT_GE(result.accuracy, 90.0F) << "FP32 should reach ≥90% accuracy";
}

/// Test TF32 reaches high accuracy.
TEST(MixedPrecisionTest, TF32HighAccuracy) {
  auto result = run_training(PrecisionMode::kTF32, 50);
  EXPECT_GE(result.accuracy, 90.0F) << "TF32 should reach ≥90% accuracy";
}

/// Test FP16 reaches high accuracy.
TEST(MixedPrecisionTest, FP16HighAccuracy) {
  auto result = run_training(PrecisionMode::kFP16, 50);
  EXPECT_GE(result.accuracy, 90.0F) << "FP16 should reach ≥90% accuracy";
}

/// Test that FP16 values saturate correctly at the FP16 range boundary.
TEST(MixedPrecisionTest, FP16Saturation) {
  // FP16 max is 65504.  Values beyond should saturate to inf.
  constexpr int N = 4;
  std::vector<float> h_in = {65504.0F, 100000.0F, -65504.0F, -100000.0F};

  float* d_fp32;
  __half* d_fp16;
  float* d_out;
  CUDA_ASSERT(cudaMalloc(&d_fp32, N * sizeof(float)));
  CUDA_ASSERT(cudaMalloc(&d_fp16, N * sizeof(__half)));
  CUDA_ASSERT(cudaMalloc(&d_out, N * sizeof(float)));

  CUDA_ASSERT(cudaMemcpy(d_fp32, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  fp32_to_fp16(d_fp32, d_fp16, N);
  fp16_to_fp32(d_fp16, d_out, N);
  CUDA_ASSERT(cudaDeviceSynchronize());

  std::vector<float> h_out(N);
  CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 65504 should round-trip exactly.
  EXPECT_FLOAT_EQ(h_out[0], 65504.0F);
  // 100000 overflows FP16 → inf.
  EXPECT_TRUE(std::isinf(h_out[1])) << "100000 should overflow to inf in FP16";
  // Negative boundary.
  EXPECT_FLOAT_EQ(h_out[2], -65504.0F);
  EXPECT_TRUE(std::isinf(h_out[3])) << "-100000 should overflow to -inf in FP16";

  CUDA_ASSERT(cudaFree(d_fp32));
  CUDA_ASSERT(cudaFree(d_fp16));
  CUDA_ASSERT(cudaFree(d_out));
}
