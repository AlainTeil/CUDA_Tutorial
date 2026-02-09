/**
 * @file transpose_test.cu
 * @brief Unit tests for Lesson 10 — Matrix Transpose.
 */

#include <gtest/gtest.h>

#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

constexpr int kTile = 32;

__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) out[col * rows + row] = in[row * cols + col];
}

__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
  __shared__ float tile[kTile][kTile + 1];
  int col = blockIdx.x * kTile + threadIdx.x;
  int row = blockIdx.y * kTile + threadIdx.y;
  if (row < rows && col < cols) tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
  __syncthreads();
  int out_col = blockIdx.y * kTile + threadIdx.x;
  int out_row = blockIdx.x * kTile + threadIdx.y;
  if (out_row < cols && out_col < rows)
    out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
}

struct MatDim {
  int rows, cols;
};

// CPU reference transpose
static std::vector<float> cpu_transpose(const std::vector<float>& in, int rows, int cols) {
  std::vector<float> out(in.size());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      out[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
  return out;
}

class TransposeTest : public ::testing::TestWithParam<MatDim> {};

TEST_P(TransposeTest, NaiveMatchesCPU) {
  auto [rows, cols] = GetParam();
  size_t total = static_cast<size_t>(rows) * cols;
  std::vector<float> h_in(total);
  for (size_t i = 0; i < total; ++i) h_in[i] = static_cast<float>(i);

  auto expected = cpu_transpose(h_in, rows, cols);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_naive<<<blocks, threads>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(total);
  CUDA_CHECK(cudaMemcpy(result.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result, expected);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

TEST_P(TransposeTest, TiledMatchesCPU) {
  auto [rows, cols] = GetParam();
  size_t total = static_cast<size_t>(rows) * cols;
  std::vector<float> h_in(total);
  for (size_t i = 0; i < total; ++i) h_in[i] = static_cast<float>(i);

  auto expected = cpu_transpose(h_in, rows, cols);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<<<blocks, threads>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(total);
  CUDA_CHECK(cudaMemcpy(result.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result, expected);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

TEST_P(TransposeTest, DoubleTransposeIsIdentity) {
  auto [rows, cols] = GetParam();
  size_t total = static_cast<size_t>(rows) * cols;
  std::vector<float> h_in(total);
  for (size_t i = 0; i < total; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_tmp, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks1((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<<<blocks1, threads>>>(d_in, d_tmp, rows, cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Transpose back: now transposed is cols×rows
  dim3 blocks2((rows + kTile - 1) / kTile, (cols + kTile - 1) / kTile);
  transpose_tiled<<<blocks2, threads>>>(d_tmp, d_out, cols, rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(total);
  CUDA_CHECK(cudaMemcpy(result.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result, h_in);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
}

INSTANTIATE_TEST_SUITE_P(Dims, TransposeTest,
                         ::testing::Values(MatDim{1, 1}, MatDim{32, 32}, MatDim{33, 31},
                                           MatDim{64, 128}, MatDim{100, 200}));
