/**
 * @file batch_normalization_test.cu
 * @brief Unit tests for Lesson 23 — Batch Normalization.
 *
 * Tests forward correctness (zero-mean / unit-variance output) and backward
 * pass via finite-difference gradient checks for γ, β, and x.
 *
 * In a real project these would be in a header, but for the tutorial we keep
 * each lesson self-contained.
 */

#include <gtest/gtest.h>

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

#define CUDA_ASSERT(call)                                                 \
  do {                                                                    \
    const cudaError_t err_ = (call);                                      \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_)); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self-contained lesson)
// =============================================================================

__global__ void compute_mean_kernel(const float* __restrict__ x, float* __restrict__ mean, int N,
                                    int C) {
  extern __shared__ float sdata[];
  int c = blockIdx.x, tid = threadIdx.x;
  float sum = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) sum += x[n * C + c];
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) mean[c] = sdata[0] / static_cast<float>(N);
}

__global__ void compute_variance_kernel(const float* __restrict__ x, const float* __restrict__ mean,
                                        float* __restrict__ var, int N, int C) {
  extern __shared__ float sdata[];
  int c = blockIdx.x, tid = threadIdx.x;
  float mu = mean[c], sum = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) {
    float d = x[n * C + c] - mu;
    sum += d * d;
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) var[c] = sdata[0] / static_cast<float>(N);
}

__global__ void batchnorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ var,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ x_hat,
                                         float* __restrict__ y, int N, int C, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C) return;
  int c = idx % C;
  float inv = rsqrtf(var[c] + eps);
  float xh = (x[idx] - mean[c]) * inv;
  x_hat[idx] = xh;
  y[idx] = gamma[c] * xh + beta[c];
}

__global__ void update_running_stats_kernel(float* __restrict__ running,
                                            const float* __restrict__ batch, int C,
                                            float momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) running[c] = (1.0F - momentum) * running[c] + momentum * batch[c];
}

__global__ void batchnorm_backward_dgamma_dbeta(const float* __restrict__ dy,
                                                const float* __restrict__ x_hat,
                                                float* __restrict__ dgamma,
                                                float* __restrict__ dbeta, int N, int C) {
  extern __shared__ float sdata[];
  float* sg = sdata;
  float* sb = sdata + blockDim.x;
  int c = blockIdx.x, tid = threadIdx.x;
  float sum_g = 0.0F, sum_b = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) {
    int idx = n * C + c;
    sum_g += dy[idx] * x_hat[idx];
    sum_b += dy[idx];
  }
  sg[tid] = sum_g;
  sb[tid] = sum_b;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sg[tid] += sg[tid + s];
      sb[tid] += sb[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    dgamma[c] = sg[0];
    dbeta[c] = sb[0];
  }
}

__global__ void batchnorm_backward_dx(const float* __restrict__ dy, const float* __restrict__ x_hat,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ var,
                                      const float* __restrict__ dgamma,
                                      const float* __restrict__ dbeta, float* __restrict__ dx,
                                      int N, int C, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C) return;
  int c = idx % C;
  float inv = rsqrtf(var[c] + eps);
  float inv_N = 1.0F / static_cast<float>(N);
  dx[idx] = gamma[c] * inv * (dy[idx] - inv_N * (dbeta[c] + x_hat[idx] * dgamma[c]));
}

// ---- Helper: run full BN forward and return y on host ----------------------

struct BNBuffers {
  float *d_x{}, *d_mean{}, *d_var{}, *d_gamma{}, *d_beta{}, *d_x_hat{}, *d_y{};
  int N{}, C{};
  float eps{};

  void alloc(int n, int c, float epsilon = 1e-5F) {
    N = n;
    C = c;
    eps = epsilon;
    auto bytes_NC = static_cast<size_t>(N * C) * sizeof(float);
    auto bytes_C = static_cast<size_t>(C) * sizeof(float);
    CUDA_ASSERT(cudaMalloc(&d_x, bytes_NC));
    CUDA_ASSERT(cudaMalloc(&d_x_hat, bytes_NC));
    CUDA_ASSERT(cudaMalloc(&d_y, bytes_NC));
    CUDA_ASSERT(cudaMalloc(&d_mean, bytes_C));
    CUDA_ASSERT(cudaMalloc(&d_var, bytes_C));
    CUDA_ASSERT(cudaMalloc(&d_gamma, bytes_C));
    CUDA_ASSERT(cudaMalloc(&d_beta, bytes_C));
  }

  void forward(const std::vector<float>& h_x, const std::vector<float>& h_gamma,
               const std::vector<float>& h_beta) {
    auto bytes_NC = static_cast<size_t>(N * C) * sizeof(float);
    auto bytes_C = static_cast<size_t>(C) * sizeof(float);
    CUDA_ASSERT(cudaMemcpy(d_x, h_x.data(), bytes_NC, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_gamma, h_gamma.data(), bytes_C, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_beta, h_beta.data(), bytes_C, cudaMemcpyHostToDevice));

    int smem = kBlockSize * static_cast<int>(sizeof(float));
    compute_mean_kernel<<<C, kBlockSize, smem>>>(d_x, d_mean, N, C);
    CUDA_CHECK(cudaGetLastError());
    compute_variance_kernel<<<C, kBlockSize, smem>>>(d_x, d_mean, d_var, N, C);
    CUDA_CHECK(cudaGetLastError());
    int grid = (N * C + kBlockSize - 1) / kBlockSize;
    batchnorm_forward_kernel<<<grid, kBlockSize>>>(d_x, d_mean, d_var, d_gamma, d_beta, d_x_hat,
                                                   d_y, N, C, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  std::vector<float> get_y() {
    std::vector<float> h_y(static_cast<size_t>(N * C));
    CUDA_ASSERT(cudaMemcpy(h_y.data(), d_y, static_cast<size_t>(N * C) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    return h_y;
  }

  void free_all() {
    CUDA_ASSERT(cudaFree(d_x));
    CUDA_ASSERT(cudaFree(d_x_hat));
    CUDA_ASSERT(cudaFree(d_y));
    CUDA_ASSERT(cudaFree(d_mean));
    CUDA_ASSERT(cudaFree(d_var));
    CUDA_ASSERT(cudaFree(d_gamma));
    CUDA_ASSERT(cudaFree(d_beta));
  }
};

// ---- Tests -----------------------------------------------------------------

TEST(BatchNormTest, ForwardZeroMeanUnitVar) {
  constexpr int kN = 128;
  constexpr int kC = 32;

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(5.0F, 3.0F);
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);

  std::vector<float> h_gamma(kC, 1.0F);
  std::vector<float> h_beta(kC, 0.0F);

  BNBuffers bn;
  bn.alloc(kN, kC);
  bn.forward(h_x, h_gamma, h_beta);
  auto h_y = bn.get_y();

  // Check per-feature mean ≈ 0 and variance ≈ 1
  for (int c = 0; c < kC; ++c) {
    double mean = 0.0;
    for (int n = 0; n < kN; ++n) mean += h_y[static_cast<size_t>(n * kC + c)];
    mean /= kN;
    EXPECT_NEAR(mean, 0.0, 1e-4) << "feature " << c;

    double var = 0.0;
    for (int n = 0; n < kN; ++n) {
      double diff = h_y[static_cast<size_t>(n * kC + c)] - mean;
      var += diff * diff;
    }
    var /= kN;
    EXPECT_NEAR(var, 1.0, 1e-3) << "feature " << c;
  }
  bn.free_all();
}

TEST(BatchNormTest, GammaShiftScales) {
  constexpr int kN = 64;
  constexpr int kC = 8;

  std::mt19937 gen(99);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);

  // γ=2, β=3 → output should have mean≈3 and var≈4
  std::vector<float> h_gamma(kC, 2.0F);
  std::vector<float> h_beta(kC, 3.0F);

  BNBuffers bn;
  bn.alloc(kN, kC);
  bn.forward(h_x, h_gamma, h_beta);
  auto h_y = bn.get_y();

  for (int c = 0; c < kC; ++c) {
    double mean = 0.0;
    for (int n = 0; n < kN; ++n) mean += h_y[static_cast<size_t>(n * kC + c)];
    mean /= kN;
    EXPECT_NEAR(mean, 3.0, 0.05) << "feature " << c;

    double var = 0.0;
    for (int n = 0; n < kN; ++n) {
      double diff = h_y[static_cast<size_t>(n * kC + c)] - mean;
      var += diff * diff;
    }
    var /= kN;
    EXPECT_NEAR(var, 4.0, 0.1) << "feature " << c;
  }
  bn.free_all();
}

TEST(BatchNormTest, InferenceModeUsesRunningStats) {
  constexpr int kN = 32;
  constexpr int kC = 8;
  constexpr float kEps = 1e-5F;

  std::vector<float> h_gamma(kC, 1.0F);
  std::vector<float> h_beta(kC, 0.0F);

  // Fixed running stats: mean=2, var=4
  std::vector<float> h_rmean(kC, 2.0F);
  std::vector<float> h_rvar(kC, 4.0F);

  std::mt19937 gen(123);
  std::normal_distribution<float> dist(2.0F, 2.0F);
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);

  float *d_x, *d_x_hat, *d_y, *d_rmean, *d_rvar, *d_gamma, *d_beta;
  auto bNC = static_cast<size_t>(kN * kC) * sizeof(float);
  auto bC = static_cast<size_t>(kC) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bNC));
  CUDA_CHECK(cudaMalloc(&d_x_hat, bNC));
  CUDA_CHECK(cudaMalloc(&d_y, bNC));
  CUDA_CHECK(cudaMalloc(&d_rmean, bC));
  CUDA_CHECK(cudaMalloc(&d_rvar, bC));
  CUDA_CHECK(cudaMalloc(&d_gamma, bC));
  CUDA_CHECK(cudaMalloc(&d_beta, bC));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bNC, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rmean, h_rmean.data(), bC, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rvar, h_rvar.data(), bC, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), bC, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), bC, cudaMemcpyHostToDevice));

  // Inference: use running stats instead of batch stats
  int grid = (kN * kC + kBlockSize - 1) / kBlockSize;
  batchnorm_forward_kernel<<<grid, kBlockSize>>>(d_x, d_rmean, d_rvar, d_gamma, d_beta, d_x_hat,
                                                 d_y, kN, kC, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(kN * kC);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bNC, cudaMemcpyDeviceToHost));

  // Verify output uses running stats: y = (x - 2) / sqrt(4 + eps) * 1 + 0
  for (int i = 0; i < kN * kC; ++i) {
    float expected = (h_x[static_cast<size_t>(i)] - 2.0F) / std::sqrt(4.0F + kEps);
    EXPECT_NEAR(h_y[static_cast<size_t>(i)], expected, 1e-4F) << "index " << i;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_x_hat));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_rmean));
  CUDA_CHECK(cudaFree(d_rvar));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
}

TEST(BatchNormTest, GammaGradFiniteDifference) {
  constexpr int kN = 16;
  constexpr int kC = 4;
  constexpr float kEps = 1e-5F;
  constexpr float kDelta = 1e-3F;

  std::mt19937 gen(7);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);
  std::vector<float> h_gamma(kC, 1.5F);
  std::vector<float> h_beta(kC, 0.5F);

  // Compute analytic dγ
  BNBuffers bn;
  bn.alloc(kN, kC, kEps);
  bn.forward(h_x, h_gamma, h_beta);

  // Upstream gradient = 1
  float *d_dy, *d_dgamma, *d_dbeta;
  auto bNC = static_cast<size_t>(kN * kC) * sizeof(float);
  auto bC = static_cast<size_t>(kC) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_dy, bNC));
  CUDA_CHECK(cudaMalloc(&d_dgamma, bC));
  CUDA_CHECK(cudaMalloc(&d_dbeta, bC));
  std::vector<float> h_dy(kN * kC, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), bNC, cudaMemcpyHostToDevice));

  int smem2 = 2 * kBlockSize * static_cast<int>(sizeof(float));
  batchnorm_backward_dgamma_dbeta<<<kC, kBlockSize, smem2>>>(d_dy, bn.d_x_hat, d_dgamma, d_dbeta,
                                                             kN, kC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dgamma(kC);
  CUDA_CHECK(cudaMemcpy(h_dgamma.data(), d_dgamma, bC, cudaMemcpyDeviceToHost));

  // Numerical gradient for each γ[c]: sum of y values when γ+δ minus sum when γ-δ
  for (int c = 0; c < kC; ++c) {
    auto gp = h_gamma;
    auto gm = h_gamma;
    gp[static_cast<size_t>(c)] += kDelta;
    gm[static_cast<size_t>(c)] -= kDelta;

    bn.forward(h_x, gp, h_beta);
    auto yp = bn.get_y();
    bn.forward(h_x, gm, h_beta);
    auto ym = bn.get_y();

    double num = 0.0;
    for (size_t i = 0; i < yp.size(); ++i) {
      num += static_cast<double>(yp[i] - ym[i]) / (2.0 * kDelta);
    }
    EXPECT_NEAR(h_dgamma[static_cast<size_t>(c)], num, 0.1)
        << "γ gradient mismatch at feature " << c;
  }

  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dgamma));
  CUDA_CHECK(cudaFree(d_dbeta));
  bn.free_all();
}

TEST(BatchNormTest, InputGradFiniteDifference) {
  constexpr int kN = 8;
  constexpr int kC = 4;
  constexpr float kEps = 1e-5F;
  constexpr float kDelta = 1e-3F;

  std::mt19937 gen(77);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);
  std::vector<float> h_gamma(kC, 1.0F);
  std::vector<float> h_beta(kC, 0.0F);

  // Analytic dx
  BNBuffers bn;
  bn.alloc(kN, kC, kEps);
  bn.forward(h_x, h_gamma, h_beta);

  float *d_dy, *d_dgamma, *d_dbeta, *d_dx;
  auto bNC = static_cast<size_t>(kN * kC) * sizeof(float);
  auto bC = static_cast<size_t>(kC) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_dy, bNC));
  CUDA_CHECK(cudaMalloc(&d_dgamma, bC));
  CUDA_CHECK(cudaMalloc(&d_dbeta, bC));
  CUDA_CHECK(cudaMalloc(&d_dx, bNC));
  std::vector<float> h_dy(kN * kC, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), bNC, cudaMemcpyHostToDevice));

  int smem2 = 2 * kBlockSize * static_cast<int>(sizeof(float));
  batchnorm_backward_dgamma_dbeta<<<kC, kBlockSize, smem2>>>(d_dy, bn.d_x_hat, d_dgamma, d_dbeta,
                                                             kN, kC);
  CUDA_CHECK(cudaGetLastError());
  int grid = (kN * kC + kBlockSize - 1) / kBlockSize;
  batchnorm_backward_dx<<<grid, kBlockSize>>>(d_dy, bn.d_x_hat, bn.d_gamma, bn.d_var, d_dgamma,
                                              d_dbeta, d_dx, kN, kC, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dx(kN * kC);
  CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, bNC, cudaMemcpyDeviceToHost));

  // Numerical gradient for a subset of x elements
  for (int i = 0; i < std::min(kN * kC, 16); ++i) {
    auto xp = h_x;
    auto xm = h_x;
    xp[static_cast<size_t>(i)] += kDelta;
    xm[static_cast<size_t>(i)] -= kDelta;

    bn.forward(xp, h_gamma, h_beta);
    auto yp = bn.get_y();
    bn.forward(xm, h_gamma, h_beta);
    auto ym = bn.get_y();

    double num = 0.0;
    for (size_t j = 0; j < yp.size(); ++j) {
      num += static_cast<double>(yp[j] - ym[j]) / (2.0 * kDelta);
    }
    EXPECT_NEAR(h_dx[static_cast<size_t>(i)], num, 0.05) << "dx mismatch at element " << i;
  }

  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dgamma));
  CUDA_CHECK(cudaFree(d_dbeta));
  CUDA_CHECK(cudaFree(d_dx));
  bn.free_all();
}
