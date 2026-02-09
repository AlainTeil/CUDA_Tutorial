/**
 * @file thread_hierarchy_test.cu
 * @brief Unit tests for Lesson 05 — Thread Hierarchy.
 */

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

// ---- Kernels ----------------------------------------------------------------

__global__ void fill_1d(int* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = idx;
}

__global__ void fill_2d(int* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    out[row * cols + col] = row * cols + col;
  }
}

__global__ void fill_3d(int* out, int depth, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < cols && y < rows && z < depth) {
    out[z * rows * cols + y * cols + x] = z * rows * cols + y * cols + x;
  }
}

// ---- 1-D tests --------------------------------------------------------------

class Fill1DTest : public ::testing::TestWithParam<int> {};

TEST_P(Fill1DTest, MatchesIota) {
  int n = GetParam();
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, static_cast<size_t>(n) * sizeof(int)));

  fill_1d<<<(n + 255) / 256, 256>>>(d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h(static_cast<size_t>(n));
  CUDA_CHECK(
      cudaMemcpy(h.data(), d_out, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> expected(static_cast<size_t>(n));
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(h, expected);

  CUDA_CHECK(cudaFree(d_out));
}

INSTANTIATE_TEST_SUITE_P(Sizes, Fill1DTest, ::testing::Values(1, 31, 256, 1023, 4096));

// ---- 2-D tests --------------------------------------------------------------

struct Dim2D {
  int rows, cols;
};

class Fill2DTest : public ::testing::TestWithParam<Dim2D> {};

TEST_P(Fill2DTest, UniqueLinearIndex) {
  auto [rows, cols] = GetParam();
  int total = rows * cols;

  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, static_cast<size_t>(total) * sizeof(int)));

  dim3 threads(16, 16);
  dim3 blocks(static_cast<unsigned>((cols + 15) / 16), static_cast<unsigned>((rows + 15) / 16));
  fill_2d<<<blocks, threads>>>(d_out, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h(static_cast<size_t>(total));
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, static_cast<size_t>(total) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) {
    EXPECT_EQ(h[static_cast<size_t>(i)], i) << "at linear index " << i;
  }

  CUDA_CHECK(cudaFree(d_out));
}

INSTANTIATE_TEST_SUITE_P(Dims, Fill2DTest,
                         ::testing::Values(Dim2D{1, 1}, Dim2D{4, 4}, Dim2D{33, 17}, Dim2D{64, 64},
                                           Dim2D{100, 100}));

// ---- 3-D tests --------------------------------------------------------------

struct Dim3D {
  int depth, rows, cols;
};

class Fill3DTest : public ::testing::TestWithParam<Dim3D> {};

TEST_P(Fill3DTest, UniqueLinearIndex) {
  auto [depth, rows, cols] = GetParam();
  int total = depth * rows * cols;

  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, static_cast<size_t>(total) * sizeof(int)));

  dim3 threads(8, 4, 2);
  dim3 blocks(static_cast<unsigned>((cols + 7) / 8), static_cast<unsigned>((rows + 3) / 4),
              static_cast<unsigned>((depth + 1) / 2));
  fill_3d<<<blocks, threads>>>(d_out, depth, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h(static_cast<size_t>(total));
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, static_cast<size_t>(total) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) {
    EXPECT_EQ(h[static_cast<size_t>(i)], i) << "at linear index " << i;
  }

  CUDA_CHECK(cudaFree(d_out));
}

INSTANTIATE_TEST_SUITE_P(Dims, Fill3DTest,
                         ::testing::Values(Dim3D{1, 1, 1}, Dim3D{2, 3, 4}, Dim3D{4, 8, 16},
                                           Dim3D{5, 7, 13}));
