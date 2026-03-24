/**
 * @file cooperative_groups_test.cu
 * @brief Unit tests for Lesson 26 — Cooperative Groups.
 *
 * Tests tile reduction, CG builtin reduce, coalesced filtering, and
 * grid-level cooperative reduction against CPU reference sums.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace cg = cooperative_groups;

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

__global__ void reduce_tile_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? in[idx] : 0.0F;
  for (int offset = tile.size() / 2; offset > 0; offset /= 2) val += tile.shfl_down(val, offset);
  __shared__ float ws[kBlockSize / 32];
  if (tile.thread_rank() == 0) ws[threadIdx.x / 32] = val;
  block.sync();
  int nw = blockDim.x / 32;
  if (threadIdx.x < static_cast<unsigned>(nw)) {
    val = ws[threadIdx.x];
    auto ft = cg::tiled_partition<32>(block);
    for (int offset = ft.size() / 2; offset > 0; offset /= 2) val += ft.shfl_down(val, offset);
    if (threadIdx.x == 0) atomicAdd(out, val);
  }
}

__global__ void reduce_cg_builtin_kernel(const float* __restrict__ in, float* __restrict__ out,
                                         int n) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? in[idx] : 0.0F;
  float warp_sum = cg::reduce(tile, val, cg::plus<float>());
  __shared__ float ws[kBlockSize / 32];
  if (tile.thread_rank() == 0) ws[threadIdx.x / 32] = warp_sum;
  block.sync();
  if (threadIdx.x < static_cast<unsigned>(blockDim.x / 32)) {
    auto ft = cg::tiled_partition<32>(block);
    float bs = cg::reduce(ft, ws[threadIdx.x], cg::plus<float>());
    if (threadIdx.x == 0) atomicAdd(out, bs);
  }
}

__global__ void coalesced_filter_kernel(const float* __restrict__ in, int* __restrict__ count,
                                        int n, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (in[idx] > threshold) {
    auto active = cg::coalesced_threads();
    if (active.thread_rank() == 0) atomicAdd(count, static_cast<int>(active.size()));
  }
}

__global__ void grid_reduce_kernel(const float* __restrict__ in, float* __restrict__ partial_sums,
                                   float* __restrict__ out, int n) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float val = 0.0F;
  for (int i = idx; i < n; i += stride) val += in[i];
  float ws2 = cg::reduce(tile, val, cg::plus<float>());
  __shared__ float ws[kBlockSize / 32];
  if (tile.thread_rank() == 0) ws[threadIdx.x / 32] = ws2;
  block.sync();
  float bs = 0.0F;
  int n_warps = blockDim.x / 32;
  if (threadIdx.x < 32u) {
    float lane_val = (threadIdx.x < static_cast<unsigned>(n_warps)) ? ws[threadIdx.x] : 0.0F;
    bs = cg::reduce(tile, lane_val, cg::plus<float>());
  }
  if (threadIdx.x == 0) partial_sums[blockIdx.x] = bs;
  grid.sync();
  if (blockIdx.x == 0) {
    float pv = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(gridDim.x);
         i += static_cast<int>(blockDim.x))
      pv += partial_sums[i];
    float warp_pv = cg::reduce(tile, pv, cg::plus<float>());
    if (tile.thread_rank() == 0) ws[threadIdx.x / 32] = warp_pv;
    block.sync();
    float total = 0.0F;
    if (threadIdx.x < 32u) {
      float lv = (threadIdx.x < static_cast<unsigned>(n_warps)) ? ws[threadIdx.x] : 0.0F;
      total = cg::reduce(tile, lv, cg::plus<float>());
    }
    if (threadIdx.x == 0) *out = total;
  }
}

// ---- Helpers ----------------------------------------------------------------

static float gpu_reduce_tile(const std::vector<float>& h_in) {
  int n = static_cast<int>(h_in.size());
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  int grid = (n + kBlockSize - 1) / kBlockSize;
  reduce_tile_kernel<<<grid, kBlockSize>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float result = 0.0F;
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return result;
}

static float gpu_reduce_builtin(const std::vector<float>& h_in) {
  int n = static_cast<int>(h_in.size());
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  int grid = (n + kBlockSize - 1) / kBlockSize;
  reduce_cg_builtin_kernel<<<grid, kBlockSize>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float result = 0.0F;
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return result;
}

// ---- Tests -----------------------------------------------------------------

class CoopGroupsSizeTest : public ::testing::TestWithParam<int> {};

TEST_P(CoopGroupsSizeTest, TileReductionMatchesReference) {
  int n = GetParam();
  std::vector<float> h_in(static_cast<size_t>(n));
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : h_in) v = dist(gen);

  double cpu_sum = 0.0;
  for (auto v : h_in) cpu_sum += v;

  float gpu_sum = gpu_reduce_tile(h_in);
  EXPECT_NEAR(gpu_sum, cpu_sum, std::abs(cpu_sum) * 1e-3 + 1e-2) << "n=" << n;
}

TEST_P(CoopGroupsSizeTest, BuiltinReductionMatchesReference) {
  int n = GetParam();
  std::vector<float> h_in(static_cast<size_t>(n));
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : h_in) v = dist(gen);

  double cpu_sum = 0.0;
  for (auto v : h_in) cpu_sum += v;

  float gpu_sum = gpu_reduce_builtin(h_in);
  EXPECT_NEAR(gpu_sum, cpu_sum, std::abs(cpu_sum) * 1e-3 + 1e-2) << "n=" << n;
}

INSTANTIATE_TEST_SUITE_P(Sizes, CoopGroupsSizeTest,
                         ::testing::Values(1, 31, 32, 256, 1024, 1 << 16, 1 << 20));

TEST(CoopGroupsTest, CoalescedFilterCount) {
  constexpr int kN = 10000;
  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> h_in(kN);
  for (auto& v : h_in) v = dist(gen);

  int cpu_count = 0;
  for (auto v : h_in) {
    if (v > 0.5F) ++cpu_count;
  }

  float* d_in;
  int* d_count;
  CUDA_CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  int grid = (kN + kBlockSize - 1) / kBlockSize;
  coalesced_filter_kernel<<<grid, kBlockSize>>>(d_in, d_count, kN, 0.5F);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int gpu_count = 0;
  CUDA_CHECK(cudaMemcpy(&gpu_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(gpu_count, cpu_count);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_count));
}

TEST(CoopGroupsTest, GridReduction) {
  // Check if cooperative launch is supported
  int supports_coop = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, 0));
  if (!supports_coop) {
    GTEST_SKIP() << "Device does not support cooperative launch";
  }

  constexpr int kN = 1 << 18;
  std::vector<float> h_in(kN, 1.0F);
  double cpu_sum = static_cast<double>(kN);

  float *d_in, *d_out, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

  int num_sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  int max_bpsm = 0;
  CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, grid_reduce_kernel, kBlockSize, 0));
  int max_grid = num_sms * max_bpsm;
  int need = (kN + kBlockSize - 1) / kBlockSize;
  int grid_size = (need < max_grid) ? need : max_grid;

  CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

  int n_val = kN;
  void* args[] = {&d_in, &d_partial, &d_out, &n_val};
  CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(grid_reduce_kernel),
                                         dim3(grid_size), dim3(kBlockSize), args));
  CUDA_CHECK(cudaDeviceSynchronize());

  float result = 0.0F;
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(result, cpu_sum, 1.0) << "Grid reduction should match CPU sum";

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_partial));
}
