/**
 * @file dense_layer_test.cu
 * @brief Unit tests for Lesson 12 — Dense Layer (forward & backward).
 *
 * Uses numerical gradient checking (finite differences) to verify backward.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

constexpr int kTile = 16;

// ---- Kernels (same as lesson) -----------------------------------------------

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) out[col * rows + row] = in[row * cols + col];
}

__global__ void add_bias(float* Y, const float* b, int batch, int out_dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < out_dim) Y[row * out_dim + col] += b[col];
}

__global__ void bias_grad(const float* dY, float* db, int batch, int out_dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < out_dim) {
    float sum = 0.0F;
    for (int i = 0; i < batch; ++i) sum += dY[i * out_dim + col];
    db[col] = sum;
  }
}

// ---- Host wrappers ----------------------------------------------------------

static void gpu_matmul(const float* dA, const float* dB, float* dC, int M, int N, int K) {
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  matmul_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
}

static void gpu_transpose(const float* d_in, float* d_out, int rows, int cols) {
  dim3 threads(kTile, kTile);
  dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_kernel<<<blocks, threads>>>(d_in, d_out, rows, cols);
}

static void dense_forward(const float* dX, const float* dW, const float* db, float* dY, int batch,
                          int in_dim, int out_dim) {
  gpu_matmul(dX, dW, dY, batch, out_dim, in_dim);
  dim3 threads(kTile, kTile);
  dim3 blocks((out_dim + kTile - 1) / kTile, (batch + kTile - 1) / kTile);
  add_bias<<<blocks, threads>>>(dY, db, batch, out_dim);
}

static void dense_backward(const float* dX, const float* dW, const float* dY_grad, float* dX_grad,
                           float* dW_grad, float* db_grad, int batch, int in_dim, int out_dim) {
  float *dWt, *dXt;
  cudaMalloc(&dWt, static_cast<size_t>(out_dim) * in_dim * sizeof(float));
  cudaMalloc(&dXt, static_cast<size_t>(in_dim) * batch * sizeof(float));

  gpu_transpose(dW, dWt, in_dim, out_dim);
  gpu_matmul(dY_grad, dWt, dX_grad, batch, in_dim, out_dim);

  gpu_transpose(dX, dXt, batch, in_dim);
  gpu_matmul(dXt, dY_grad, dW_grad, in_dim, out_dim, batch);

  bias_grad<<<(out_dim + 255) / 256, 256>>>(dY_grad, db_grad, batch, out_dim);

  cudaFree(dWt);
  cudaFree(dXt);
}

// ---- CPU reference ----------------------------------------------------------

static void cpu_dense_forward(const float* X, const float* W, const float* b, float* Y, int batch,
                              int in_dim, int out_dim) {
  for (int i = 0; i < batch; ++i)
    for (int j = 0; j < out_dim; ++j) {
      float s = b[j];
      for (int k = 0; k < in_dim; ++k) s += X[i * in_dim + k] * W[k * out_dim + j];
      Y[i * out_dim + j] = s;
    }
}

// ---- Tests ------------------------------------------------------------------

TEST(DenseLayerTest, ForwardMatchesCPU) {
  constexpr int kBatch = 4, kIn = 16, kOut = 8;

  std::vector<float> hX(kBatch * kIn), hW(kIn * kOut), hb(kOut);
  srand(42);
  for (auto& v : hX) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hW) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hb) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  // CPU
  std::vector<float> hY_ref(kBatch * kOut);
  cpu_dense_forward(hX.data(), hW.data(), hb.data(), hY_ref.data(), kBatch, kIn, kOut);

  // GPU
  float *dX, *dW, *db, *dY;
  CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, hb.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, hY_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice));

  dense_forward(dX, dW, db, dY, kBatch, kIn, kOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hY(hY_ref.size());
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < hY.size(); ++i) EXPECT_NEAR(hY[i], hY_ref[i], 1e-3F) << "at index " << i;

  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dY));
}

TEST(DenseLayerTest, BackwardGradientCheck) {
  constexpr int kBatch = 2, kIn = 4, kOut = 3;
  constexpr float kEps = 1e-3F;

  std::vector<float> hX(kBatch * kIn), hW(kIn * kOut), hb(kOut);
  srand(99);
  for (auto& v : hX) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hW) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hb) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  // dY_grad = all ones (gradient of sum(Y))
  std::vector<float> hDY(kBatch * kOut, 1.0F);

  // GPU forward + backward
  float *dX, *dW, *db, *dY, *dDY, *dXg, *dWg, *dBg;
  CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, hb.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, hDY.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dDY, hDY.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dXg, hX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWg, hW.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dBg, hb.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDY, hDY.data(), hDY.size() * sizeof(float), cudaMemcpyHostToDevice));

  dense_backward(dX, dW, dDY, dXg, dWg, dBg, kBatch, kIn, kOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hWg(hW.size()), hBg(hb.size());
  CUDA_CHECK(cudaMemcpy(hWg.data(), dWg, hWg.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hBg.data(), dBg, hBg.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check for W
  auto loss_fn = [&](const std::vector<float>& W_test) -> float {
    std::vector<float> Y(kBatch * kOut);
    cpu_dense_forward(hX.data(), W_test.data(), hb.data(), Y.data(), kBatch, kIn, kOut);
    float sum = 0;
    for (auto v : Y) sum += v;
    return sum;
  };

  for (size_t i = 0; i < hW.size(); ++i) {
    auto Wp = hW;
    auto Wm = hW;
    Wp[i] += kEps;
    Wm[i] -= kEps;
    float num_grad = (loss_fn(Wp) - loss_fn(Wm)) / (2 * kEps);
    EXPECT_NEAR(hWg[i], num_grad, 1e-2F) << "dW mismatch at index " << i;
  }

  // Numerical gradient check for b
  for (size_t i = 0; i < hb.size(); ++i) {
    auto bp = hb;
    auto bm = hb;
    bp[i] += kEps;
    bm[i] -= kEps;
    std::vector<float> Yp(kBatch * kOut), Ym(kBatch * kOut);
    cpu_dense_forward(hX.data(), hW.data(), bp.data(), Yp.data(), kBatch, kIn, kOut);
    cpu_dense_forward(hX.data(), hW.data(), bm.data(), Ym.data(), kBatch, kIn, kOut);
    float fp = 0, fm = 0;
    for (auto v : Yp) fp += v;
    for (auto v : Ym) fm += v;
    float num_grad = (fp - fm) / (2 * kEps);
    EXPECT_NEAR(hBg[i], num_grad, 1e-2F) << "db mismatch at index " << i;
  }

  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dY));
  CUDA_CHECK(cudaFree(dDY));
  CUDA_CHECK(cudaFree(dXg));
  CUDA_CHECK(cudaFree(dWg));
  CUDA_CHECK(cudaFree(dBg));
}
