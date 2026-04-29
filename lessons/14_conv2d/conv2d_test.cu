/**
 * @file conv2d_test.cu
 * @brief Unit tests for Lesson 14 — 2D Convolution.
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

__global__ void conv2d_direct(const float* in, const float* kernel, float* out, int H, int W,
                              int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ < OH && oc < OW) {
    float sum = 0.0F;
    for (int kr = 0; kr < KH; ++kr)
      for (int kc = 0; kc < KW; ++kc) sum += in[(or_ + kr) * W + (oc + kc)] * kernel[kr * KW + kc];
    out[or_ * OW + oc] = sum;
  }
}

// ----- im2col + tiled GEMM (mirror of conv2d.cu) -----
__global__ void im2col_kernel(const float* in, float* col, int H, int W, int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  int total = OH * OW;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int or_ = idx / OW;
    int oc = idx % OW;
    for (int kr = 0; kr < KH; ++kr)
      for (int kc = 0; kc < KW; ++kc)
        col[(kr * KW + kc) * total + idx] = in[(or_ + kr) * W + (oc + kc)];
  }
}

constexpr int kTile = 16;

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0F;
  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    int a_col = t * kTile + threadIdx.x;
    int b_row = t * kTile + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0F;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0F;
    __syncthreads();
    for (int k = 0; k < kTile; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

static void launch_im2col_gemm(const float* d_in, const float* d_kernel, float* d_out, int H, int W,
                               int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  const int K = KH * KW;
  const int N = OH * OW;
  float* d_col = nullptr;
  CUDA_CHECK(cudaMalloc(&d_col, static_cast<size_t>(K) * N * sizeof(float)));
  constexpr int kBlock = 256;
  im2col_kernel<<<(N + kBlock - 1) / kBlock, kBlock>>>(d_in, d_col, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (1 + kTile - 1) / kTile);
  matmul_tiled<<<blocks, threads>>>(d_kernel, d_col, d_out, /*M=*/1, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_col));
}

// ----- Backward kernels (mirror of conv2d.cu) -----
__global__ void conv2d_backward_dkernel(const float* __restrict__ in,
                                        const float* __restrict__ dout, float* __restrict__ dkernel,
                                        int H, int W, int KH, int KW) {
  int kc = blockIdx.x * blockDim.x + threadIdx.x;
  int kr = blockIdx.y * blockDim.y + threadIdx.y;
  if (kr >= KH || kc >= KW) return;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  float acc = 0.0F;
  for (int r = 0; r < OH; ++r)
    for (int c = 0; c < OW; ++c) acc += dout[r * OW + c] * in[(r + kr) * W + (c + kc)];
  dkernel[kr * KW + kc] = acc;
}

__global__ void conv2d_backward_dx(const float* __restrict__ dout, const float* __restrict__ kernel,
                                   float* din, int H, int W, int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ >= OH || oc >= OW) return;
  const float g = dout[or_ * OW + oc];
  for (int kr = 0; kr < KH; ++kr)
    for (int kc = 0; kc < KW; ++kc)
      atomicAdd(&din[(or_ + kr) * W + (oc + kc)], g * kernel[kr * KW + kc]);
}

static void cpu_conv2d(const float* in, const float* k, float* out, int H, int W, int KH, int KW) {
  int OH = H - KH + 1, OW = W - KW + 1;
  for (int r = 0; r < OH; ++r)
    for (int c = 0; c < OW; ++c) {
      float s = 0;
      for (int kr = 0; kr < KH; ++kr)
        for (int kc = 0; kc < KW; ++kc) s += in[(r + kr) * W + (c + kc)] * k[kr * KW + kc];
      out[r * OW + c] = s;
    }
}

struct ConvParams {
  int H, W, KH, KW;
};

class Conv2DTest : public ::testing::TestWithParam<ConvParams> {};

TEST_P(Conv2DTest, DirectMatchesCPU) {
  auto [H, W, KH, KW] = GetParam();
  int OH = H - KH + 1, OW = W - KW + 1;

  std::vector<float> h_in(static_cast<size_t>(H) * W), h_k(static_cast<size_t>(KH) * KW);
  srand(42);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));
  for (auto& v : h_k) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));

  std::vector<float> h_ref(static_cast<size_t>(OH) * OW);
  cpu_conv2d(h_in.data(), h_k.data(), h_ref.data(), H, W, KH, KW);

  float *d_in, *d_k, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);
  conv2d_direct<<<blocks, threads>>>(d_in, d_k, d_out, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(h_ref.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < h_out.size(); ++i) {
    EXPECT_NEAR(h_out[i], h_ref[i], 1e-4F) << "at " << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_out));
}

INSTANTIATE_TEST_SUITE_P(Dims, Conv2DTest,
                         ::testing::Values(ConvParams{5, 5, 3, 3}, ConvParams{8, 8, 3, 3},
                                           ConvParams{16, 16, 5, 5}, ConvParams{32, 32, 3, 3},
                                           ConvParams{7, 9, 3, 3}));

TEST(Conv2DTest, IdentityKernel) {
  // A kernel with 1 at center and 0 elsewhere should produce the centre crop
  constexpr int H = 8, W = 8, K = 3;
  constexpr int OH = H - K + 1, OW = W - K + 1;

  std::vector<float> h_in(H * W);
  for (int i = 0; i < H * W; ++i) h_in[static_cast<size_t>(i)] = static_cast<float>(i);

  std::vector<float> h_k(K * K, 0.0F);
  h_k[4] = 1.0F;  // center of 3x3

  float *d_in, *d_k, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);
  conv2d_direct<<<blocks, threads>>>(d_in, d_k, d_out, H, W, K, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(static_cast<size_t>(OH) * OW);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < OH; ++r)
    for (int c = 0; c < OW; ++c)
      EXPECT_FLOAT_EQ(h_out[static_cast<size_t>(r) * OW + c],
                      h_in[static_cast<size_t>(r + 1) * W + (c + 1)]);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_out));
}

// =============================================================================
// im2col + GEMM tests
// =============================================================================

class Conv2DIm2colTest : public ::testing::TestWithParam<ConvParams> {};

TEST_P(Conv2DIm2colTest, Im2colGemmMatchesCPU) {
  auto [H, W, KH, KW] = GetParam();
  int OH = H - KH + 1, OW = W - KW + 1;

  std::vector<float> h_in(static_cast<size_t>(H) * W), h_k(static_cast<size_t>(KH) * KW);
  srand(123);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : h_k) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  std::vector<float> h_ref(static_cast<size_t>(OH) * OW);
  cpu_conv2d(h_in.data(), h_k.data(), h_ref.data(), H, W, KH, KW);

  float *d_in, *d_k, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  launch_im2col_gemm(d_in, d_k, d_out, H, W, KH, KW);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(h_ref.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Tolerance scales with K = KH*KW (the GEMM accumulation depth).
  const int K = KH * KW;
  const float tol_factor = 1e-5F * static_cast<float>(K);
  for (size_t i = 0; i < h_out.size(); ++i) {
    EXPECT_NEAR(h_out[i], h_ref[i], 1e-4F + tol_factor * std::abs(h_ref[i])) << "at " << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_out));
}

TEST_P(Conv2DIm2colTest, Im2colGemmAgreesWithDirect) {
  auto [H, W, KH, KW] = GetParam();
  int OH = H - KH + 1, OW = W - KW + 1;

  std::vector<float> h_in(static_cast<size_t>(H) * W), h_k(static_cast<size_t>(KH) * KW);
  srand(7);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : h_k) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  float *d_in, *d_k, *d_direct, *d_gemm;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_direct, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gemm, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);
  conv2d_direct<<<blocks, threads>>>(d_in, d_k, d_direct, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());

  launch_im2col_gemm(d_in, d_k, d_gemm, H, W, KH, KW);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_direct(static_cast<size_t>(OH) * OW), h_gemm(static_cast<size_t>(OH) * OW);
  CUDA_CHECK(cudaMemcpy(h_direct.data(), d_direct, h_direct.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_gemm.data(), d_gemm, h_gemm.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // The two paths should agree to near float precision, since both perform
  // the same KH*KW multiply-adds in essentially the same order.
  const float tol_factor = 1e-5F * static_cast<float>(KH * KW);
  for (size_t i = 0; i < h_direct.size(); ++i) {
    EXPECT_NEAR(h_gemm[i], h_direct[i], 1e-4F + tol_factor * std::abs(h_direct[i])) << "at " << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_direct));
  CUDA_CHECK(cudaFree(d_gemm));
}

INSTANTIATE_TEST_SUITE_P(Dims, Conv2DIm2colTest,
                         ::testing::Values(ConvParams{5, 5, 3, 3}, ConvParams{8, 8, 3, 3},
                                           ConvParams{16, 16, 5, 5}, ConvParams{32, 32, 3, 3},
                                           ConvParams{7, 9, 3, 3}));

// =============================================================================
// Backward-pass tests (finite-difference verification)
// =============================================================================
//
// Loss L = sum(out[r][c]) so that dL/dout = 1 everywhere.  We then check the
// analytic dkernel and dx kernels against the central-difference numerical
// gradient of the same scalar loss.

namespace {

float scalar_loss(const std::vector<float>& v) {
  float s = 0.0F;
  for (float x : v) s += x;
  return s;
}

std::vector<float> run_forward(const std::vector<float>& h_in, const std::vector<float>& h_k, int H,
                               int W, int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  std::vector<float> h_out(static_cast<size_t>(OH) * OW);
  cpu_conv2d(h_in.data(), h_k.data(), h_out.data(), H, W, KH, KW);
  return h_out;
}

}  // namespace

TEST(Conv2DBackwardTest, KernelGradMatchesFiniteDiff) {
  constexpr int H = 6, W = 6, KH = 3, KW = 3;
  constexpr int OH = H - KH + 1, OW = W - KW + 1;
  constexpr float kEps = 1e-3F;

  std::vector<float> h_in(H * W), h_k(KH * KW);
  srand(11);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : h_k) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  // ---- Analytic dkernel via the GPU kernel ----
  float *d_in, *d_k, *d_dout, *d_dk;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> h_dout(static_cast<size_t>(OH) * OW, 1.0F);
  CUDA_CHECK(
      cudaMemcpy(d_dout, h_dout.data(), h_dout.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 k_threads(KW, KH);
  conv2d_backward_dkernel<<<dim3(1, 1), k_threads>>>(d_in, d_dout, d_dk, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dk(KH * KW);
  CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, h_dk.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- Numerical dkernel via central differences on the CPU forward ----
  for (int i = 0; i < KH * KW; ++i) {
    auto kp = h_k;
    auto km = h_k;
    kp[static_cast<size_t>(i)] += kEps;
    km[static_cast<size_t>(i)] -= kEps;
    float lp = scalar_loss(run_forward(h_in, kp, H, W, KH, KW));
    float lm = scalar_loss(run_forward(h_in, km, H, W, KH, KW));
    float num = (lp - lm) / (2.0F * kEps);
    EXPECT_NEAR(h_dk[static_cast<size_t>(i)], num, 1e-2F) << "kernel weight " << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_dk));
}

TEST(Conv2DBackwardTest, InputGradMatchesFiniteDiff) {
  constexpr int H = 6, W = 6, KH = 3, KW = 3;
  constexpr int OH = H - KH + 1, OW = W - KW + 1;
  constexpr float kEps = 1e-3F;

  std::vector<float> h_in(H * W), h_k(KH * KW);
  srand(13);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : h_k) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  float *d_in, *d_k, *d_dout, *d_dx;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> h_dout(static_cast<size_t>(OH) * OW, 1.0F);
  CUDA_CHECK(
      cudaMemcpy(d_dout, h_dout.data(), h_dout.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dx, 0, h_in.size() * sizeof(float)));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);
  conv2d_backward_dx<<<blocks, threads>>>(d_dout, d_k, d_dx, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dx(h_in.size());
  CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, h_dx.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < H * W; ++i) {
    auto xp = h_in;
    auto xm = h_in;
    xp[static_cast<size_t>(i)] += kEps;
    xm[static_cast<size_t>(i)] -= kEps;
    float lp = scalar_loss(run_forward(xp, h_k, H, W, KH, KW));
    float lm = scalar_loss(run_forward(xm, h_k, H, W, KH, KW));
    float num = (lp - lm) / (2.0F * kEps);
    EXPECT_NEAR(h_dx[static_cast<size_t>(i)], num, 1e-2F) << "input pixel " << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_dx));
}
