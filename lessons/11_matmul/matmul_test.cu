/**
 * @file matmul_test.cu
 * @brief Unit tests for Lesson 11 — Matrix Multiplication.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

constexpr int kTile = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float sum = 0.0F;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];
  int col = blockIdx.x * kTile + threadIdx.x;
  int row = blockIdx.y * kTile + threadIdx.y;
  float sum = 0.0F;
  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    int a_col = t * kTile + threadIdx.x;
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0F;
    int b_row = t * kTile + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0F;
    __syncthreads();
    for (int k = 0; k < kTile; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = sum;
}

static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

struct MatmulDim {
  int M, N, K;
};

class MatmulTest : public ::testing::TestWithParam<MatmulDim> {};

TEST_P(MatmulTest, NaiveMatchesCPU) {
  auto [M, N, K] = GetParam();

  std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N),
      hC_ref(static_cast<size_t>(M) * N);

  srand(42);
  for (auto& v : hA) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hB) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  matmul_naive<<<blocks, threads>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC(hC_ref.size());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < hC.size(); ++i) {
    EXPECT_NEAR(hC[i], hC_ref[i], 1e-3F) << "at index " << i;
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
}

TEST_P(MatmulTest, TiledMatchesCPU) {
  auto [M, N, K] = GetParam();

  std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N),
      hC_ref(static_cast<size_t>(M) * N);

  srand(42);
  for (auto& v : hA) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hB) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  matmul_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC(hC_ref.size());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < hC.size(); ++i) {
    EXPECT_NEAR(hC[i], hC_ref[i], 1e-3F) << "at index " << i;
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
}

INSTANTIATE_TEST_SUITE_P(Dims, MatmulTest,
                         ::testing::Values(MatmulDim{1, 1, 1}, MatmulDim{16, 16, 16},
                                           MatmulDim{33, 17, 25}, MatmulDim{64, 128, 64},
                                           MatmulDim{128, 256, 64}));

TEST(MatmulTest, IdentityMatrix) {
  constexpr int kN = 32;
  std::vector<float> hA(kN * kN, 0.0F), hI(kN * kN, 0.0F);
  srand(123);
  for (auto& v : hA) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));
  for (int i = 0; i < kN; ++i) hI[static_cast<size_t>(i) * kN + i] = 1.0F;

  float *dA, *dI, *dC;
  size_t bytes = kN * kN * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dI, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dI, hI.data(), bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((kN + kTile - 1) / kTile, (kN + kTile - 1) / kTile);
  matmul_tiled<<<blocks, threads>>>(dA, dI, dC, kN, kN, kN);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC(kN * kN);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN * kN; ++i)
    EXPECT_NEAR(hC[static_cast<size_t>(i)], hA[static_cast<size_t>(i)], 1e-5F);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dI));
  CUDA_CHECK(cudaFree(dC));
}
