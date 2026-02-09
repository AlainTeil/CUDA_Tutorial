/**
 * @file activations_test.cu
 * @brief Unit tests for Lesson 13 — Activation Functions.
 *
 * Tests forward correctness and backward via finite-difference gradient check.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

// ---- Kernels ----------------------------------------------------------------

__global__ void relu_forward(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(0.0F, in[i]);
}
__global__ void relu_backward(const float* in, const float* go, float* gi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) gi[i] = (in[i] > 0.0F) ? go[i] : 0.0F;
}

__global__ void sigmoid_forward(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0F / (1.0F + expf(-in[i]));
}
__global__ void sigmoid_backward(const float* out, const float* go, float* gi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = out[i];
    gi[i] = go[i] * s * (1.0F - s);
  }
}

__global__ void tanh_forward(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tanhf(in[i]);
}
__global__ void tanh_backward(const float* out, const float* go, float* gi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float t = out[i];
    gi[i] = go[i] * (1.0F - t * t);
  }
}

__global__ void softmax_forward(const float* in, float* out, int cols) {
  extern __shared__ float sd[];
  int row = blockIdx.x, tid = threadIdx.x;
  const float* ri = in + row * cols;
  float* ro = out + row * cols;
  float mx = -1e30F;
  for (int c = tid; c < cols; c += blockDim.x) mx = fmaxf(mx, ri[c]);
  sd[tid] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sd[tid] = fmaxf(sd[tid], sd[tid + s]);
    __syncthreads();
  }
  float rm = sd[0];
  float ls = 0;
  for (int c = tid; c < cols; c += blockDim.x) {
    float e = expf(ri[c] - rm);
    ro[c] = e;
    ls += e;
  }
  sd[tid] = ls;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sd[tid] += sd[tid + s];
    __syncthreads();
  }
  float tot = sd[0];
  for (int c = tid; c < cols; c += blockDim.x) ro[c] /= tot;
}

// ---- Helpers ----------------------------------------------------------------

static std::vector<float> gpu_apply(void (*fwd)(const float*, float*, int),
                                    const std::vector<float>& input) {
  int n = static_cast<int>(input.size());
  float *d_in, *d_out;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  fwd<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
  cudaDeviceSynchronize();
  std::vector<float> result(static_cast<size_t>(n));
  cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  return result;
}

// ---- ReLU Tests -------------------------------------------------------------

TEST(ActivationsTest, ReluForward) {
  std::vector<float> in = {-3, -1, 0, 1, 3};
  auto out = gpu_apply(relu_forward, in);
  EXPECT_FLOAT_EQ(out[0], 0.0F);
  EXPECT_FLOAT_EQ(out[1], 0.0F);
  EXPECT_FLOAT_EQ(out[2], 0.0F);
  EXPECT_FLOAT_EQ(out[3], 1.0F);
  EXPECT_FLOAT_EQ(out[4], 3.0F);
}

TEST(ActivationsTest, ReluBackwardFiniteDiff) {
  constexpr int kN = 64;
  constexpr float kEps = 1e-3F;
  std::vector<float> in(kN);
  for (int i = 0; i < kN; ++i) in[static_cast<size_t>(i)] = static_cast<float>(i) / kN - 0.5F;

  // Analytic grad
  std::vector<float> go(kN, 1.0F);
  float *d_in, *d_out, *d_go, *d_gi;
  cudaMalloc(&d_in, kN * sizeof(float));
  cudaMalloc(&d_out, kN * sizeof(float));
  cudaMalloc(&d_go, kN * sizeof(float));
  cudaMalloc(&d_gi, kN * sizeof(float));
  cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_go, go.data(), kN * sizeof(float), cudaMemcpyHostToDevice);

  relu_forward<<<1, 256>>>(d_in, d_out, kN);
  relu_backward<<<1, 256>>>(d_in, d_go, d_gi, kN);
  cudaDeviceSynchronize();

  std::vector<float> gi(kN);
  cudaMemcpy(gi.data(), d_gi, kN * sizeof(float), cudaMemcpyDeviceToHost);

  // Numerical grad
  for (int i = 0; i < kN; ++i) {
    auto ip = in;
    auto im = in;
    ip[static_cast<size_t>(i)] += kEps;
    im[static_cast<size_t>(i)] -= kEps;
    auto op = gpu_apply(relu_forward, ip);
    auto om = gpu_apply(relu_forward, im);
    float num = 0;
    for (int j = 0; j < kN; ++j)
      num += (op[static_cast<size_t>(j)] - om[static_cast<size_t>(j)]) / (2 * kEps);
    // Skip at exactly 0 (non-differentiable)
    if (std::abs(in[static_cast<size_t>(i)]) > kEps)
      EXPECT_NEAR(gi[static_cast<size_t>(i)], num, 0.1F) << "at " << i;
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_go);
  cudaFree(d_gi);
}

// ---- Sigmoid Tests ----------------------------------------------------------

TEST(ActivationsTest, SigmoidForward) {
  std::vector<float> in = {0.0F};
  auto out = gpu_apply(sigmoid_forward, in);
  EXPECT_NEAR(out[0], 0.5F, 1e-5F);
}

TEST(ActivationsTest, SigmoidRange) {
  std::vector<float> in = {-10, -1, 0, 1, 10};
  auto out = gpu_apply(sigmoid_forward, in);
  for (auto v : out) {
    EXPECT_GT(v, 0.0F);
    EXPECT_LT(v, 1.0F);
  }
}

// ---- Tanh Tests -------------------------------------------------------------

TEST(ActivationsTest, TanhForward) {
  std::vector<float> in = {0.0F};
  auto out = gpu_apply(tanh_forward, in);
  EXPECT_NEAR(out[0], 0.0F, 1e-5F);
}

TEST(ActivationsTest, TanhRange) {
  std::vector<float> in = {-10, -1, 0, 1, 10};
  auto out = gpu_apply(tanh_forward, in);
  for (auto v : out) {
    EXPECT_GE(v, -1.0F);
    EXPECT_LE(v, 1.0F);
  }
}

// ---- Softmax Tests ----------------------------------------------------------

TEST(ActivationsTest, SoftmaxSumsToOne) {
  constexpr int kCols = 10;
  constexpr int kRows = 4;
  std::vector<float> in(kRows * kCols);
  srand(77);
  for (auto& v : in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) * 4.0F - 2.0F;

  float *d_in, *d_out;
  cudaMalloc(&d_in, in.size() * sizeof(float));
  cudaMalloc(&d_out, in.size() * sizeof(float));
  cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);

  int smem = 256 * static_cast<int>(sizeof(float));
  softmax_forward<<<kRows, 256, smem>>>(d_in, d_out, kCols);
  cudaDeviceSynchronize();

  std::vector<float> out(in.size());
  cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int r = 0; r < kRows; ++r) {
    float sum = 0;
    for (int c = 0; c < kCols; ++c) {
      float v = out[static_cast<size_t>(r) * kCols + c];
      EXPECT_GE(v, 0.0F);
      sum += v;
    }
    EXPECT_NEAR(sum, 1.0F, 1e-5F) << "row " << r;
  }

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST(ActivationsTest, SoftmaxAllEqual) {
  constexpr int kCols = 5;
  std::vector<float> in(kCols, 1.0F);

  float *d_in, *d_out;
  cudaMalloc(&d_in, kCols * sizeof(float));
  cudaMalloc(&d_out, kCols * sizeof(float));
  cudaMemcpy(d_in, in.data(), kCols * sizeof(float), cudaMemcpyHostToDevice);

  int smem = 256 * static_cast<int>(sizeof(float));
  softmax_forward<<<1, 256, smem>>>(d_in, d_out, kCols);
  cudaDeviceSynchronize();

  std::vector<float> out(kCols);
  cudaMemcpy(out.data(), d_out, kCols * sizeof(float), cudaMemcpyDeviceToHost);

  for (int c = 0; c < kCols; ++c) EXPECT_NEAR(out[static_cast<size_t>(c)], 0.2F, 1e-5F);

  cudaFree(d_in);
  cudaFree(d_out);
}
