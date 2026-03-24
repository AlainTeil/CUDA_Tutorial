/**
 * @file dropout_test.cu
 * @brief Unit tests for Lesson 24 — Dropout Regularization.
 *
 * Tests drop rate, inverted scaling, backward mask consistency, inference
 * identity, and seed reproducibility/difference.
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

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self-contained lesson)
// =============================================================================

__device__ __forceinline__ unsigned int hash_index(unsigned long long seed, int idx) {
  unsigned long long h = seed ^ (static_cast<unsigned long long>(idx) * 2654435761ULL);
  h ^= h >> 17;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 31;
  h *= 0x94d049bb133111ebULL;
  h ^= h >> 32;
  return static_cast<unsigned int>(h);
}

__device__ __forceinline__ float hash_to_uniform(unsigned int h) {
  return static_cast<float>(h & 0x7FFFFFU) / static_cast<float>(0x800000U);
}

__global__ void dropout_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                       float* __restrict__ mask, int n, float p,
                                       unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  unsigned int h = hash_index(seed, idx);
  float u = hash_to_uniform(h);
  float keep = (u >= p) ? 1.0F : 0.0F;
  float scale = 1.0F / (1.0F - p);
  mask[idx] = keep;
  y[idx] = x[idx] * keep * scale;
}

__global__ void dropout_backward_kernel(const float* __restrict__ grad_out,
                                        const float* __restrict__ mask, float* __restrict__ grad_in,
                                        int n, float p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float scale = 1.0F / (1.0F - p);
  grad_in[idx] = grad_out[idx] * mask[idx] * scale;
}

__global__ void dropout_inference_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx];
}

// ---- Helpers ----------------------------------------------------------------

struct DropResult {
  std::vector<float> y;
  std::vector<float> mask;
};

static DropResult run_dropout_forward(const std::vector<float>& h_x, float p,
                                      unsigned long long seed) {
  int n = static_cast<int>(h_x.size());
  float *d_x, *d_y, *d_mask;
  auto bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_mask, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  int grid = (n + kBlockSize - 1) / kBlockSize;
  dropout_forward_kernel<<<grid, kBlockSize>>>(d_x, d_y, d_mask, n, p, seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  DropResult res;
  res.y.resize(static_cast<size_t>(n));
  res.mask.resize(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(res.y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(res.mask.data(), d_mask, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_mask));
  return res;
}

// ---- Tests -----------------------------------------------------------------

TEST(DropoutTest, ApproximateDropRate) {
  constexpr int kN = 100000;
  constexpr float kP = 0.3F;

  std::vector<float> h_x(kN, 1.0F);
  auto res = run_dropout_forward(h_x, kP, 42ULL);

  int zeros = 0;
  for (auto m : res.mask) {
    if (m == 0.0F) ++zeros;
  }
  float drop_rate = static_cast<float>(zeros) / static_cast<float>(kN);
  EXPECT_NEAR(drop_rate, kP, 0.02F) << "Drop rate should be approximately p=" << kP;
}

TEST(DropoutTest, InvertedScaling) {
  constexpr int kN = 10000;
  constexpr float kP = 0.4F;

  std::vector<float> h_x(kN, 3.0F);  // constant input
  auto res = run_dropout_forward(h_x, kP, 99ULL);

  float scale = 1.0F / (1.0F - kP);
  for (int i = 0; i < kN; ++i) {
    if (res.mask[static_cast<size_t>(i)] == 1.0F) {
      EXPECT_FLOAT_EQ(res.y[static_cast<size_t>(i)], 3.0F * scale)
          << "Kept element should be scaled by 1/(1-p)";
    } else {
      EXPECT_FLOAT_EQ(res.y[static_cast<size_t>(i)], 0.0F) << "Dropped element should be exactly 0";
    }
  }
}

TEST(DropoutTest, BackwardMaskConsistency) {
  constexpr int kN = 4096;
  constexpr float kP = 0.5F;

  std::vector<float> h_x(kN);
  for (int i = 0; i < kN; ++i) h_x[static_cast<size_t>(i)] = static_cast<float>(i);

  auto fwd = run_dropout_forward(h_x, kP, 77ULL);

  // Run backward with grad_out = 1
  float *d_go, *d_gi, *d_mask;
  auto bytes = static_cast<size_t>(kN) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_go, bytes));
  CUDA_CHECK(cudaMalloc(&d_gi, bytes));
  CUDA_CHECK(cudaMalloc(&d_mask, bytes));

  std::vector<float> h_go(kN, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_go, h_go.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, fwd.mask.data(), bytes, cudaMemcpyHostToDevice));

  int grid = (kN + kBlockSize - 1) / kBlockSize;
  dropout_backward_kernel<<<grid, kBlockSize>>>(d_go, d_mask, d_gi, kN, kP);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_gi(kN);
  CUDA_CHECK(cudaMemcpy(h_gi.data(), d_gi, bytes, cudaMemcpyDeviceToHost));

  float scale = 1.0F / (1.0F - kP);
  for (int i = 0; i < kN; ++i) {
    if (fwd.mask[static_cast<size_t>(i)] == 0.0F) {
      EXPECT_FLOAT_EQ(h_gi[static_cast<size_t>(i)], 0.0F)
          << "Gradient should be 0 where activation was dropped";
    } else {
      EXPECT_FLOAT_EQ(h_gi[static_cast<size_t>(i)], scale)
          << "Gradient should be scaled by 1/(1-p) where kept";
    }
  }

  CUDA_CHECK(cudaFree(d_go));
  CUDA_CHECK(cudaFree(d_gi));
  CUDA_CHECK(cudaFree(d_mask));
}

TEST(DropoutTest, InferenceModeIsIdentity) {
  constexpr int kN = 512;

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0F, 5.0F);
  std::vector<float> h_x(kN);
  for (auto& v : h_x) v = dist(gen);

  float *d_x, *d_y;
  auto bytes = static_cast<size_t>(kN) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  int grid = (kN + kBlockSize - 1) / kBlockSize;
  dropout_inference_kernel<<<grid, kBlockSize>>>(d_x, d_y, kN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(kN);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(h_y[static_cast<size_t>(i)], h_x[static_cast<size_t>(i)])
        << "Inference dropout should be identity";
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
}

TEST(DropoutTest, DifferentSeedsDifferentMasks) {
  constexpr int kN = 4096;
  constexpr float kP = 0.5F;

  std::vector<float> h_x(kN, 1.0F);
  auto res1 = run_dropout_forward(h_x, kP, 111ULL);
  auto res2 = run_dropout_forward(h_x, kP, 222ULL);

  int differences = 0;
  for (int i = 0; i < kN; ++i) {
    if (res1.mask[static_cast<size_t>(i)] != res2.mask[static_cast<size_t>(i)]) ++differences;
  }
  // With p=0.5 and 4096 elements, ~50% should differ
  EXPECT_GT(differences, kN / 4) << "Different seeds should produce substantially different masks";
}

TEST(DropoutTest, SameSeedSameMask) {
  constexpr int kN = 1024;
  constexpr float kP = 0.5F;

  std::vector<float> h_x(kN, 1.0F);
  auto res1 = run_dropout_forward(h_x, kP, 42ULL);
  auto res2 = run_dropout_forward(h_x, kP, 42ULL);

  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(res1.mask[static_cast<size_t>(i)], res2.mask[static_cast<size_t>(i)])
        << "Same seed should produce identical masks";
  }
}
