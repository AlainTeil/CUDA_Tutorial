/**
 * @file conv2d_test.cu
 * @brief Unit tests for Lesson 14 — 2D Convolution.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
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
