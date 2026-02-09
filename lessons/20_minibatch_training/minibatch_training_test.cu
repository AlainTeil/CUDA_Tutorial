/**
 * @file minibatch_training_test.cu
 * @brief Unit tests for Lesson 20 — Mini-Batch Training with cuBLAS.
 *
 * Verifies:
 *  1. cuBLAS-based dense forward matches a CPU reference.
 *  2. Batched log-softmax produces valid log-probabilities.
 *  3. Bias-gradient (column-sum) is correct.
 *  4. Gradient-averaged dW matches CPU reference.
 *  5. Loss decreases over mini-batch training epochs.
 *  6. Final accuracy exceeds 90% on linearly-separable synthetic data.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// =============================================================================
// Error-checking macros
// =============================================================================

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) FAIL() << "CUDA error: " << cudaGetErrorString(err_); \
  } while (0)

#define CUDA_ASSERT(call)                                                 \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_)); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

#define CUBLAS_CHECK(call)                                                                 \
  do {                                                                                     \
    cublasStatus_t st_ = (call);                                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) FAIL() << "cuBLAS error: " << static_cast<int>(st_); \
  } while (0)

#define CUBLAS_ASSERT(call)                                                   \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error: code %d\n", static_cast<int>(st_)); \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

// =============================================================================
// Kernel definitions (self-contained for single-TU compilation)
// =============================================================================

static void gemm_rm(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                    const float* B, float beta, float* C) {
  CUBLAS_ASSERT(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

static void gemm_rm_AT(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  CUBLAS_ASSERT(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

__global__ void relu_fwd(const float* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (in[i] > 0.0F) ? in[i] : 0.0F;
}

__global__ void relu_bwd(const float* pre_act, const float* grad_out, float* grad_in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad_in[i] = (pre_act[i] > 0.0F) ? grad_out[i] : 0.0F;
}

__global__ void add_bias(float* Y, const float* bias, int B, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * D) {
    int j = idx % D;
    Y[idx] += bias[j];
  }
}

__global__ void batch_log_softmax(const float* logits, float* log_sm, int C) {
  extern __shared__ float sdata[];
  int b = blockIdx.x;
  int tid = threadIdx.x;
  const float* row = logits + b * C;
  float val = (tid < C) ? row[tid] : -1e30F;
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float mx = sdata[0];
  __syncthreads();
  float ev = (tid < C) ? expf(val - mx) : 0.0F;
  sdata[tid] = ev;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float lse = logf(sdata[0]);
  if (tid < C) log_sm[b * C + tid] = (val - mx) - lse;
}

__global__ void ce_fwd(const float* log_sm, const float* target, float* elem, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem[i] = -target[i] * log_sm[i];
}

__global__ void ce_bwd(const float* log_sm, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

__global__ void reduce_sum_all(const float* in, float* out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float acc = 0.0F;
  for (int i = tid; i < N; i += blockDim.x) acc += in[i];
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) *out = sdata[0];
}

__global__ void col_sum(const float* in, float* out, int B, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) {
    float s = 0.0F;
    for (int b = 0; b < B; ++b) s += in[b * D + j];
    out[j] = s;
  }
}

__global__ void scale_kernel(float* x, float scale, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] *= scale;
}

__global__ void sgd_update(float* param, const float* grad, float lr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) param[i] -= lr * grad[i];
}

// =============================================================================
// BatchMLP struct (duplicated for self-contained test TU)
// =============================================================================

struct BatchMLP {
  int in_dim{}, hid_dim{}, out_dim{}, max_batch{};
  cublasHandle_t cublas{};
  float *W1{}, *b1{}, *W2{}, *b2{};
  float *dW1{}, *db1{}, *dW2{}, *db2{};
  float *Z1{}, *A1{}, *Z2{}, *LS{};
  float *dZ2{}, *dA1{}, *dZ1{};
  float *elem_loss{}, *d_loss{};

  void alloc(int in, int hid, int out, int max_B) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    max_batch = max_B;
    CUBLAS_ASSERT(cublasCreate(&cublas));
    auto A = [](float** p, int n) {
      CUDA_ASSERT(cudaMalloc(p, static_cast<size_t>(n) * sizeof(float)));
    };
    A(&W1, in * hid);
    A(&b1, hid);
    A(&W2, hid * out);
    A(&b2, out);
    A(&dW1, in * hid);
    A(&db1, hid);
    A(&dW2, hid * out);
    A(&db2, out);
    A(&Z1, max_B * hid);
    A(&A1, max_B * hid);
    A(&Z2, max_B * out);
    A(&LS, max_B * out);
    A(&dZ2, max_B * out);
    A(&dA1, max_B * hid);
    A(&dZ1, max_B * hid);
    A(&elem_loss, max_B * out);
    CUDA_ASSERT(cudaMalloc(&d_loss, sizeof(float)));
  }

  void init_weights(unsigned seed) {
    std::mt19937 gen(seed);
    auto fill = [&](float* d, int n, float scale) {
      std::normal_distribution<float> dist(0.0F, scale);
      std::vector<float> h(static_cast<size_t>(n));
      for (auto& v : h) v = dist(gen);
      CUDA_ASSERT(
          cudaMemcpy(d, h.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    };
    fill(W1, in_dim * hid_dim, std::sqrt(2.0F / static_cast<float>(in_dim)));
    fill(b1, hid_dim, 0.01F);
    fill(W2, hid_dim * out_dim, std::sqrt(2.0F / static_cast<float>(hid_dim)));
    fill(b2, out_dim, 0.01F);
  }

  float forward(const float* d_X, const float* d_target, int B) {
    int total1 = B * hid_dim;
    int total2 = B * out_dim;
    int blk = 256;
    gemm_rm(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(total1 + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, A1, total1);
    gemm_rm(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(total2 + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    int sm_block = 1;
    while (sm_block < out_dim) sm_block <<= 1;
    batch_log_softmax<<<B, sm_block, static_cast<size_t>(sm_block) * sizeof(float)>>>(Z2, LS,
                                                                                      out_dim);
    ce_fwd<<<(total2 + blk - 1) / blk, blk>>>(LS, d_target, elem_loss, total2);
    int red_block = 256;
    reduce_sum_all<<<1, red_block, static_cast<size_t>(red_block) * sizeof(float)>>>(
        elem_loss, d_loss, total2);
    CUDA_ASSERT(cudaDeviceSynchronize());
    float h_loss;
    CUDA_ASSERT(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss / static_cast<float>(B);
  }

  void backward(const float* d_X, const float* d_target, int B) {
    int total1 = B * hid_dim;
    int total2 = B * out_dim;
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);
    ce_bwd<<<(total2 + blk - 1) / blk, blk>>>(LS, d_target, dZ2, total2);
    gemm_rm_AT(cublas, hid_dim, B, out_dim, inv_B, A1, dZ2, 0.0F, dW2);
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);
    {
      float one = 1.0F, zero = 0.0F;
      CUBLAS_ASSERT(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2,
                                out_dim, dZ2, out_dim, &zero, dA1, hid_dim));
    }
    relu_bwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, total1);
    gemm_rm_AT(cublas, in_dim, B, hid_dim, inv_B, d_X, dZ1, 0.0F, dW1);
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  void sgd_step(float lr) {
    int blk = 256;
    auto upd = [&](float* p, const float* g, int n) {
      sgd_update<<<(n + blk - 1) / blk, blk>>>(p, g, lr, n);
    };
    upd(W1, dW1, in_dim * hid_dim);
    upd(b1, db1, hid_dim);
    upd(W2, dW2, hid_dim * out_dim);
    upd(b2, db2, out_dim);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  void predict_batch(const float* d_X, int B, std::vector<int>& preds) {
    int blk = 256;
    int total1 = B * hid_dim;
    int total2 = B * out_dim;
    gemm_rm(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(total1 + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, A1, total1);
    gemm_rm(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(total2 + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    CUDA_ASSERT(cudaDeviceSynchronize());
    std::vector<float> h_z(static_cast<size_t>(total2));
    CUDA_ASSERT(cudaMemcpy(h_z.data(), Z2, static_cast<size_t>(total2) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    preds.resize(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
      auto row = h_z.begin() + b * out_dim;
      preds[static_cast<size_t>(b)] =
          static_cast<int>(std::distance(row, std::max_element(row, row + out_dim)));
    }
  }

  void free_all() {
    cublasDestroy(cublas);
    for (float* p :
         {W1, b1, W2, b2, dW1, db1, dW2, db2, Z1, A1, Z2, LS, dZ2, dA1, dZ1, elem_loss, d_loss})
      cudaFree(p);
  }
};

// =============================================================================
// Data helpers
// =============================================================================

static void generate_data(std::vector<float>& flat_samples, std::vector<int>& labels,
                          int n_per_class, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  float centres[3][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
  int N = 3 * n_per_class;
  flat_samples.resize(static_cast<size_t>(N) * 4);
  labels.resize(static_cast<size_t>(N));
  int idx = 0;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < n_per_class; ++i) {
      for (int d = 0; d < 4; ++d)
        flat_samples[static_cast<size_t>(idx) * 4 + d] = centres[c][d] + noise(gen);
      labels[static_cast<size_t>(idx)] = c;
      ++idx;
    }
}

static std::vector<float> make_one_hot(const std::vector<int>& labels, int C) {
  auto N = static_cast<int>(labels.size());
  std::vector<float> oh(static_cast<size_t>(N) * C, 0.0F);
  for (int i = 0; i < N; ++i)
    oh[static_cast<size_t>(i) * C + labels[static_cast<size_t>(i)]] = 1.0F;
  return oh;
}

// =============================================================================
// Test: cuBLAS dense forward matches CPU reference
// =============================================================================

TEST(MiniBatchTest, DenseForwardMatchesCPU) {
  constexpr int B = 4, IN = 3, OUT = 5;

  // Random input and weights on host.
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0F, 1.0F);

  std::vector<float> h_X(B * IN), h_W(IN * OUT), h_b(OUT);
  for (auto& v : h_X) v = dist(gen);
  for (auto& v : h_W) v = dist(gen);
  for (auto& v : h_b) v = dist(gen);

  // CPU reference:  Y = X · W + b
  std::vector<float> h_Y_ref(B * OUT, 0.0F);
  for (int bi = 0; bi < B; ++bi)
    for (int j = 0; j < OUT; ++j) {
      float s = h_b[static_cast<size_t>(j)];
      for (int k = 0; k < IN; ++k)
        s += h_X[static_cast<size_t>(bi) * IN + k] * h_W[static_cast<size_t>(k) * OUT + j];
      h_Y_ref[static_cast<size_t>(bi) * OUT + j] = s;
    }

  // GPU
  float *d_X, *d_W, *d_b, *d_Y;
  CUDA_CHECK(cudaMalloc(&d_X, B * IN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_W, IN * OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, B * OUT * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), B * IN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), IN * OUT * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), OUT * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  gemm_rm(handle, B, IN, OUT, 1.0F, d_X, d_W, 0.0F, d_Y);
  int total = B * OUT;
  add_bias<<<(total + 255) / 256, 256>>>(d_Y, d_b, B, OUT);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_Y_gpu(B * OUT);
  CUDA_CHECK(cudaMemcpy(h_Y_gpu.data(), d_Y, B * OUT * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < B * OUT; ++i)
    EXPECT_NEAR(h_Y_gpu[static_cast<size_t>(i)], h_Y_ref[static_cast<size_t>(i)], 1e-4F)
        << "Mismatch at index " << i;

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_Y));
}

// =============================================================================
// Test: batched log-softmax produces valid log-probabilities
// =============================================================================

TEST(MiniBatchTest, BatchLogSoftmaxValid) {
  constexpr int B = 8, C = 5;
  std::mt19937 gen(99);
  std::normal_distribution<float> dist(0.0F, 3.0F);

  std::vector<float> h_logits(B * C);
  for (auto& v : h_logits) v = dist(gen);

  float *d_logits, *d_ls;
  CUDA_CHECK(cudaMalloc(&d_logits, B * C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ls, B * C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), B * C * sizeof(float), cudaMemcpyHostToDevice));

  int block = 1;
  while (block < C) block <<= 1;
  batch_log_softmax<<<B, block, static_cast<size_t>(block) * sizeof(float)>>>(d_logits, d_ls, C);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_ls(B * C);
  CUDA_CHECK(cudaMemcpy(h_ls.data(), d_ls, B * C * sizeof(float), cudaMemcpyDeviceToHost));

  for (int b = 0; b < B; ++b) {
    // Each log_sm value should be <= 0.
    float sum_exp = 0.0F;
    for (int c = 0; c < C; ++c) {
      EXPECT_LE(h_ls[static_cast<size_t>(b) * C + c], 1e-6F)
          << "log-softmax should be ≤ 0 at b=" << b << " c=" << c;
      sum_exp += std::exp(h_ls[static_cast<size_t>(b) * C + c]);
    }
    // exp(log_softmax) should sum to 1.
    EXPECT_NEAR(sum_exp, 1.0F, 1e-5F) << "Probabilities don't sum to 1 for row " << b;
  }

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_ls));
}

// =============================================================================
// Test: column-sum (bias gradient) correctness
// =============================================================================

TEST(MiniBatchTest, ColSumCorrect) {
  constexpr int B = 6, D = 4;
  std::vector<float> h_in(B * D);
  for (int i = 0; i < B * D; ++i) h_in[static_cast<size_t>(i)] = static_cast<float>(i + 1);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, B * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), B * D * sizeof(float), cudaMemcpyHostToDevice));

  col_sum<<<1, D>>>(d_in, d_out, B, D);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(D);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, D * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference
  for (int j = 0; j < D; ++j) {
    float expected = 0.0F;
    for (int b = 0; b < B; ++b) expected += h_in[static_cast<size_t>(b) * D + j];
    EXPECT_NEAR(h_out[static_cast<size_t>(j)], expected, 1e-5F) << "col_sum mismatch at j=" << j;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

// =============================================================================
// Test: averaged dW matches CPU reference
// =============================================================================

TEST(MiniBatchTest, GradientAveragingCorrect) {
  // dW = (1/B) * A^T · dZ  where A is [B×M], dZ is [B×N], dW is [M×N].
  constexpr int B = 4, M = 3, N = 2;

  std::vector<float> h_A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // [4×3]
  std::vector<float> h_dZ = {1, 0, 0, 1, 1, 1, 0.5F, 0.5F};          // [4×2]

  float *d_A, *d_dZ, *d_dW;
  CUDA_CHECK(cudaMalloc(&d_A, B * M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dZ, B * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dW, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), B * M * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dZ, h_dZ.data(), B * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float inv_B = 1.0F / static_cast<float>(B);
  gemm_rm_AT(handle, M, B, N, inv_B, d_A, d_dZ, 0.0F, d_dW);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dW(M * N);
  CUDA_CHECK(cudaMemcpy(h_dW.data(), d_dW, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference: dW[i][j] = (1/B) * Σ_b A[b][i] * dZ[b][j]
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float expected = 0.0F;
      for (int b = 0; b < B; ++b)
        expected += h_A[static_cast<size_t>(b) * M + i] * h_dZ[static_cast<size_t>(b) * N + j];
      expected *= inv_B;
      EXPECT_NEAR(h_dW[static_cast<size_t>(i) * N + j], expected, 1e-5F)
          << "dW mismatch at (" << i << "," << j << ")";
    }

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_dZ));
  CUDA_CHECK(cudaFree(d_dW));
}

// =============================================================================
// Test: loss decreases during mini-batch training
// =============================================================================

TEST(MiniBatchTest, LossDecreases) {
  constexpr int N_PER_CLASS = 30;
  constexpr int N_TOTAL = 3 * N_PER_CLASS;
  constexpr int BATCH_SIZE = 16;
  constexpr int EPOCHS = 30;
  constexpr float LR = 0.1F;

  std::vector<float> flat_data;
  std::vector<int> labels;
  generate_data(flat_data, labels, N_PER_CLASS, 42);
  std::vector<float> one_hot = make_one_hot(labels, 3);

  BatchMLP mlp{};
  mlp.alloc(4, 16, 3, BATCH_SIZE);
  mlp.init_weights(123);

  float *d_bx, *d_bt;
  CUDA_CHECK(cudaMalloc(&d_bx, static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bt, static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float)));

  std::vector<int> indices(N_TOTAL);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(7);

  float first_epoch_loss = 0.0F;
  float last_epoch_loss = 0.0F;

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);
    float epoch_loss = 0.0F;
    int n_batches = 0;

    for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
      std::vector<float> bx(static_cast<size_t>(BATCH_SIZE) * 4);
      std::vector<float> bt(static_cast<size_t>(BATCH_SIZE) * 3);
      for (int b = 0; b < BATCH_SIZE; ++b) {
        int si = indices[static_cast<size_t>(start + b)];
        std::memcpy(&bx[static_cast<size_t>(b) * 4], &flat_data[static_cast<size_t>(si) * 4],
                    4 * sizeof(float));
        std::memcpy(&bt[static_cast<size_t>(b) * 3], &one_hot[static_cast<size_t>(si) * 3],
                    3 * sizeof(float));
      }
      CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));

      float loss = mlp.forward(d_bx, d_bt, BATCH_SIZE);
      mlp.backward(d_bx, d_bt, BATCH_SIZE);
      mlp.sgd_step(LR);
      epoch_loss += loss;
      ++n_batches;
    }
    float avg = epoch_loss / static_cast<float>(n_batches);
    if (epoch == 0) first_epoch_loss = avg;
    last_epoch_loss = avg;
  }

  EXPECT_LT(last_epoch_loss, first_epoch_loss * 0.5F)
      << "Loss should decrease significantly over training";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_bx));
  CUDA_CHECK(cudaFree(d_bt));
}

// =============================================================================
// Test: final accuracy > 90% on linearly-separable data
// =============================================================================

TEST(MiniBatchTest, HighAccuracy) {
  constexpr int N_PER_CLASS = 50;
  constexpr int N_TOTAL = 3 * N_PER_CLASS;
  constexpr int BATCH_SIZE = 32;
  constexpr int EPOCHS = 60;
  constexpr float LR = 0.1F;

  std::vector<float> flat_data;
  std::vector<int> labels;
  generate_data(flat_data, labels, N_PER_CLASS, 99);
  std::vector<float> one_hot = make_one_hot(labels, 3);

  BatchMLP mlp{};
  mlp.alloc(4, 16, 3, BATCH_SIZE);
  mlp.init_weights(456);

  float *d_bx, *d_bt;
  CUDA_CHECK(cudaMalloc(&d_bx, static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bt, static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float)));

  // Copy full data to device for prediction later.
  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(N_TOTAL) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, flat_data.data(), static_cast<size_t>(N_TOTAL) * 4 * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<int> indices(N_TOTAL);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(7);

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
      std::vector<float> bx(static_cast<size_t>(BATCH_SIZE) * 4);
      std::vector<float> bt(static_cast<size_t>(BATCH_SIZE) * 3);
      for (int b = 0; b < BATCH_SIZE; ++b) {
        int si = indices[static_cast<size_t>(start + b)];
        std::memcpy(&bx[static_cast<size_t>(b) * 4], &flat_data[static_cast<size_t>(si) * 4],
                    4 * sizeof(float));
        std::memcpy(&bt[static_cast<size_t>(b) * 3], &one_hot[static_cast<size_t>(si) * 3],
                    3 * sizeof(float));
      }
      CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));
      mlp.forward(d_bx, d_bt, BATCH_SIZE);
      mlp.backward(d_bx, d_bt, BATCH_SIZE);
      mlp.sgd_step(LR);
    }
  }

  // Evaluate accuracy on full dataset in batches.
  int correct = 0;
  int evaluated = 0;
  for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
    std::vector<int> preds;
    mlp.predict_batch(d_data + start * 4, BATCH_SIZE, preds);
    for (int b = 0; b < BATCH_SIZE; ++b)
      if (preds[static_cast<size_t>(b)] == labels[static_cast<size_t>(start + b)]) ++correct;
    evaluated += BATCH_SIZE;
  }

  double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(evaluated);
  EXPECT_GT(acc, 90.0) << "Accuracy should exceed 90% on linearly-separable data";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_bx));
  CUDA_CHECK(cudaFree(d_bt));
  CUDA_CHECK(cudaFree(d_data));
}

// =============================================================================
// Test: reduce_sum_all handles various sizes correctly
// =============================================================================

TEST(MiniBatchTest, ReduceSumAll) {
  constexpr int N = 100;
  std::vector<float> h_in(N);
  float expected = 0.0F;
  for (int i = 0; i < N; ++i) {
    h_in[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.1F;
    expected += h_in[static_cast<size_t>(i)];
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  reduce_sum_all<<<1, 256, 256 * sizeof(float)>>>(d_in, d_out, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_out;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(h_out, expected, 0.1F) << "reduce_sum_all result incorrect";

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}
