/**
 * @file scan_test.cu
 * @brief Unit tests for Lesson 09 — Prefix Sum (Exclusive Scan).
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

constexpr int kBlockSize = 256;

__global__ void blelloch_scan_block(float* data, float* block_sums, int n) {
  __shared__ float temp[kBlockSize];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  temp[tid] = (gid < n) ? data[gid] : 0.0F;
  __syncthreads();
  for (int stride = 1; stride < kBlockSize; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < kBlockSize) temp[index] += temp[index - stride];
    __syncthreads();
  }
  if (tid == 0) {
    if (block_sums) block_sums[blockIdx.x] = temp[kBlockSize - 1];
    temp[kBlockSize - 1] = 0.0F;
  }
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < kBlockSize) {
      float t = temp[index - stride];
      temp[index - stride] = temp[index];
      temp[index] += t;
    }
    __syncthreads();
  }
  if (gid < n) data[gid] = temp[tid];
}

__global__ void add_block_sums(float* data, const float* block_sums, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n && blockIdx.x > 0) data[gid] += block_sums[blockIdx.x];
}

static void gpu_exclusive_scan(float* d_data, int n) {
  int blocks = (n + kBlockSize - 1) / kBlockSize;
  if (blocks == 1) {
    blelloch_scan_block<<<1, kBlockSize>>>(d_data, nullptr, n);
    cudaDeviceSynchronize();
    return;
  }
  float* d_bs = nullptr;
  cudaMalloc(&d_bs, static_cast<size_t>(blocks) * sizeof(float));
  blelloch_scan_block<<<blocks, kBlockSize>>>(d_data, d_bs, n);
  cudaDeviceSynchronize();
  gpu_exclusive_scan(d_bs, blocks);
  add_block_sums<<<blocks, kBlockSize>>>(d_data, d_bs, n);
  cudaDeviceSynchronize();
  cudaFree(d_bs);
}

// ---- Helper -----------------------------------------------------------------

static std::vector<float> run_scan(const std::vector<float>& input) {
  int n = static_cast<int>(input.size());
  size_t bytes = input.size() * sizeof(float);
  float* d = nullptr;
  cudaMalloc(&d, bytes);
  cudaMemcpy(d, input.data(), bytes, cudaMemcpyHostToDevice);
  gpu_exclusive_scan(d, n);
  std::vector<float> result(input.size());
  cudaMemcpy(result.data(), d, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d);
  return result;
}

// ---- Tests ------------------------------------------------------------------

class ScanTest : public ::testing::TestWithParam<int> {};

TEST_P(ScanTest, MatchesCpuExclusiveScan) {
  int n = GetParam();
  std::vector<float> input(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) input[static_cast<size_t>(i)] = static_cast<float>(i % 7 + 1);

  std::vector<float> expected(static_cast<size_t>(n));
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0.0F);

  auto result = run_scan(input);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(result[static_cast<size_t>(i)], expected[static_cast<size_t>(i)],
                std::abs(expected[static_cast<size_t>(i)]) * 1e-4F + 1e-4F)
        << "at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, ScanTest, ::testing::Values(1, 2, 7, 32, 256, 512, 1000, 4096));

TEST(ScanTest, AllOnes) {
  constexpr int kN = 1024;
  std::vector<float> input(kN, 1.0F);
  auto result = run_scan(input);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(result[static_cast<size_t>(i)], static_cast<float>(i), 1e-3F);
  }
}

TEST(ScanTest, AllZeros) {
  constexpr int kN = 128;
  std::vector<float> input(kN, 0.0F);
  auto result = run_scan(input);
  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(result[static_cast<size_t>(i)], 0.0F);
  }
}
