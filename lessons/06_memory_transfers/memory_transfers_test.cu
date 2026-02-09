/**
 * @file memory_transfers_test.cu
 * @brief Unit tests for Lesson 06 — Memory Transfers.
 *
 * All three memory paths must produce the same numerical result.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

__global__ void scale_kernel(float* data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] *= factor;
}

// Helpers returning host-side results ---

static std::vector<float> run_pageable(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> h(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) h[static_cast<size_t>(i)] = static_cast<float>(i);

  float* d = nullptr;
  cudaMalloc(&d, bytes);
  cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice);
  scale_kernel<<<(n + 255) / 256, 256>>>(d, factor, n);
  cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d);
  return h;
}

static std::vector<float> run_pinned(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float* h = nullptr;
  cudaMallocHost(&h, bytes);
  for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);

  float* d = nullptr;
  cudaMalloc(&d, bytes);
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  scale_kernel<<<(n + 255) / 256, 256>>>(d, factor, n);
  cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::vector<float> result(h, h + n);
  cudaFree(d);
  cudaFreeHost(h);
  return result;
}

static std::vector<float> run_unified(int n, float factor) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float* data = nullptr;
  cudaMallocManaged(&data, bytes);
  for (int i = 0; i < n; ++i) data[i] = static_cast<float>(i);

  scale_kernel<<<(n + 255) / 256, 256>>>(data, factor, n);
  cudaDeviceSynchronize();

  std::vector<float> result(data, data + n);
  cudaFree(data);
  return result;
}

// ---- Tests ------------------------------------------------------------------

class MemoryTransferTest : public ::testing::TestWithParam<int> {};

TEST_P(MemoryTransferTest, AllPathsProduceSameResult) {
  int n = GetParam();
  constexpr float kFactor = 3.0F;

  auto pageable = run_pageable(n, kFactor);
  auto pinned = run_pinned(n, kFactor);
  auto unified = run_unified(n, kFactor);

  ASSERT_EQ(pageable.size(), static_cast<size_t>(n));
  ASSERT_EQ(pinned.size(), static_cast<size_t>(n));
  ASSERT_EQ(unified.size(), static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i) * kFactor;
    EXPECT_NEAR(pageable[static_cast<size_t>(i)], expected, 1e-5F) << "pageable mismatch at " << i;
    EXPECT_NEAR(pinned[static_cast<size_t>(i)], expected, 1e-5F) << "pinned mismatch at " << i;
    EXPECT_NEAR(unified[static_cast<size_t>(i)], expected, 1e-5F) << "unified mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, MemoryTransferTest, ::testing::Values(1, 128, 1000, 65536));
