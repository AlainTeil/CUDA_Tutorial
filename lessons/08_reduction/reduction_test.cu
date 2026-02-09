/**
 * @file reduction_test.cu
 * @brief Unit tests for Lesson 08 — Parallel Reduction.
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

__global__ void reduce_shared(const float* in, float* out, int n) {
  __shared__ float sdata[kBlockSize];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (gid < n) ? in[gid] : 0.0F;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

__device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__global__ void reduce_warp_shuffle(const float* in, float* out, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (gid < n) ? in[gid] : 0.0F;
  val = warp_reduce_sum(val);
  __shared__ float warp_sums[kBlockSize / 32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  if (lane == 0) warp_sums[wid] = val;
  __syncthreads();
  if (wid == 0) {
    val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0F;
    val = warp_reduce_sum(val);
  }
  if (threadIdx.x == 0) out[blockIdx.x] = val;
}

// Host recursive reduce
template <typename Kernel>
static float gpu_reduce(const float* d_in, int n, Kernel kernel) {
  int blocks = (n + kBlockSize - 1) / kBlockSize;
  float* d_out = nullptr;
  cudaMalloc(&d_out, static_cast<size_t>(blocks) * sizeof(float));
  kernel<<<blocks, kBlockSize>>>(d_in, d_out, n);
  cudaDeviceSynchronize();
  float result;
  if (blocks == 1) {
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    result = gpu_reduce(d_out, blocks, kernel);
  }
  cudaFree(d_out);
  return result;
}

// ---- Parameterized tests ----------------------------------------------------

class ReductionTest : public ::testing::TestWithParam<int> {};

TEST_P(ReductionTest, SharedMemReduction) {
  int n = GetParam();
  // Use constant 1.0f so the exact sum (= n) is representable in float,
  // avoiding accumulation-order differences between CPU and GPU.
  std::vector<float> h(static_cast<size_t>(n), 1.0F);

  float* d = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));

  float gpu_sum = gpu_reduce(d, n, reduce_shared);
  float cpu_sum = static_cast<float>(n);

  EXPECT_FLOAT_EQ(gpu_sum, cpu_sum);
  CUDA_CHECK(cudaFree(d));
}

TEST_P(ReductionTest, WarpShuffleReduction) {
  int n = GetParam();
  std::vector<float> h(static_cast<size_t>(n), 1.0F);

  float* d = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));

  float gpu_sum = gpu_reduce(d, n, reduce_warp_shuffle);
  float cpu_sum = static_cast<float>(n);

  EXPECT_FLOAT_EQ(gpu_sum, cpu_sum);
  CUDA_CHECK(cudaFree(d));
}

INSTANTIATE_TEST_SUITE_P(Sizes, ReductionTest,
                         ::testing::Values(1, 7, 32, 256, 1000, 4096, 1 << 16, 1 << 20));
