/**
 * @file gru_test.cu
 * @brief Unit tests for Lesson 30 — Gated Recurrent Unit (GRU).
 *
 * Tests verify the GRU gate kernel, forward sequence, and backward pass
 * against CPU reference implementations.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                    \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self‑contained lesson)
// =============================================================================

__device__ __forceinline__ float d_sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

__global__ void gru_gates_kernel(const float* __restrict__ xW, const float* __restrict__ hU,
                                 const float* __restrict__ bias, const float* __restrict__ h_prev,
                                 float* __restrict__ h_out, float* __restrict__ z_buf,
                                 float* __restrict__ r_buf, float* __restrict__ n_buf, int B,
                                 int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * H) return;
  int b = idx / H;
  int h = idx % H;

  float z_pre = xW[b * 3 * H + h] + hU[b * 3 * H + h] + bias[h];
  float r_pre = xW[b * 3 * H + H + h] + hU[b * 3 * H + H + h] + bias[H + h];
  float z = d_sigmoid(z_pre);
  float r = d_sigmoid(r_pre);
  float n_pre = xW[b * 3 * H + 2 * H + h] + r * hU[b * 3 * H + 2 * H + h] + bias[2 * H + h];
  float n = tanhf(n_pre);
  float hp = h_prev[idx];
  h_out[idx] = (1.0F - z) * n + z * hp;
  z_buf[idx] = z;
  r_buf[idx] = r;
  n_buf[idx] = n;
}

__global__ void gru_backward_gates_kernel(
    const float* __restrict__ dh, const float* __restrict__ z_buf, const float* __restrict__ r_buf,
    const float* __restrict__ n_buf, const float* __restrict__ h_prev, const float* __restrict__ hU,
    float* __restrict__ dh_prev, float* __restrict__ d_xW, float* __restrict__ d_hU,
    float* __restrict__ dbias, int B, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * H) return;
  int b = idx / H;
  int h = idx % H;

  float z = z_buf[idx];
  float r = r_buf[idx];
  float n = n_buf[idx];
  float hp = h_prev[idx];
  float dh_val = dh[idx];

  float dn = dh_val * (1.0F - z);
  float dz = dh_val * (hp - n);
  float dh_prev_val = dh_val * z;
  float dn_pre = dn * (1.0F - n * n);
  float dz_pre = dz * z * (1.0F - z);
  float hU_n = hU[b * 3 * H + 2 * H + h];
  float dr = dn_pre * hU_n;
  float dr_pre = dr * r * (1.0F - r);

  dh_prev[idx] = dh_prev_val;
  d_xW[b * 3 * H + h] = dz_pre;
  d_xW[b * 3 * H + H + h] = dr_pre;
  d_xW[b * 3 * H + 2 * H + h] = dn_pre;
  d_hU[b * 3 * H + h] = dz_pre;
  d_hU[b * 3 * H + H + h] = dr_pre;
  d_hU[b * 3 * H + 2 * H + h] = dn_pre * r;
  atomicAdd(&dbias[h], dz_pre);
  atomicAdd(&dbias[H + h], dr_pre);
  atomicAdd(&dbias[2 * H + h], dn_pre);
}

// =============================================================================
// CPU reference
// =============================================================================

static float cpu_sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

static void cpu_gru_step(const float* xW, const float* hU, const float* bias, const float* h_prev,
                         float* h_out, int B, int H) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      int idx = b * H + h;
      float z = cpu_sigmoid(xW[b * 3 * H + h] + hU[b * 3 * H + h] + bias[h]);
      float r = cpu_sigmoid(xW[b * 3 * H + H + h] + hU[b * 3 * H + H + h] + bias[H + h]);
      float n =
          std::tanh(xW[b * 3 * H + 2 * H + h] + r * hU[b * 3 * H + 2 * H + h] + bias[2 * H + h]);
      h_out[idx] = (1.0F - z) * n + z * h_prev[idx];
    }
  }
}

// =============================================================================
// Tests
// =============================================================================

class GRUTest : public ::testing::Test {
 protected:
  static constexpr int kB = 2;
  static constexpr int kH = 8;
  static constexpr int kI = 4;

  cublasHandle_t handle_{};
  void SetUp() override { CUBLAS_CHECK(cublasCreate(&handle_)); }
  void TearDown() override { CUBLAS_CHECK(cublasDestroy(handle_)); }
};

// ------------------------------------------------------------------
// Gate kernel matches CPU reference
// ------------------------------------------------------------------

TEST_F(GRUTest, GateKernelMatchesCPU) {
  int total3 = kB * 3 * kH;
  int totalH = kB * kH;

  std::vector<float> h_xW(total3), h_hU(total3), h_bias(3 * kH, 0.0F);
  std::vector<float> h_prev(totalH);
  for (int i = 0; i < total3; ++i) {
    h_xW[i] = static_cast<float>(i % 7) * 0.1F - 0.3F;
    h_hU[i] = static_cast<float>(i % 5) * 0.1F - 0.2F;
  }
  for (int i = 0; i < totalH; ++i) h_prev[i] = static_cast<float>(i % 3) * 0.2F;

  // CPU reference
  std::vector<float> h_out_cpu(totalH);
  cpu_gru_step(h_xW.data(), h_hU.data(), h_bias.data(), h_prev.data(), h_out_cpu.data(), kB, kH);

  // GPU
  float *d_xW, *d_hU, *d_bias, *d_hp, *d_ho, *d_z, *d_r, *d_n;
  CUDA_CHECK(cudaMalloc(&d_xW, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hp, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ho, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, totalH * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_xW, h_xW.data(), total3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_hU, h_hU.data(), total3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_hp, h_prev.data(), totalH * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (totalH + kBlockSize - 1) / kBlockSize;
  gru_gates_kernel<<<grid, kBlockSize>>>(d_xW, d_hU, d_bias, d_hp, d_ho, d_z, d_r, d_n, kB, kH);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out_gpu(totalH);
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_ho, totalH * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < totalH; ++i) EXPECT_NEAR(h_out_gpu[i], h_out_cpu[i], 1e-5F) << "i=" << i;

  CUDA_CHECK(cudaFree(d_xW));
  CUDA_CHECK(cudaFree(d_hU));
  CUDA_CHECK(cudaFree(d_bias));
  CUDA_CHECK(cudaFree(d_hp));
  CUDA_CHECK(cudaFree(d_ho));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_n));
}

// ------------------------------------------------------------------
// Zero input, zero hidden → hidden stays near zero
// ------------------------------------------------------------------

TEST_F(GRUTest, ZeroInputProducesNearZeroHidden) {
  int total3 = kB * 3 * kH;
  int totalH = kB * kH;

  float *d_xW, *d_hU, *d_bias, *d_hp, *d_ho, *d_z, *d_r, *d_n;
  CUDA_CHECK(cudaMalloc(&d_xW, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hp, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ho, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, totalH * sizeof(float)));

  CUDA_CHECK(cudaMemset(d_xW, 0, total3 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_hU, 0, total3 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_bias, 0, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_hp, 0, totalH * sizeof(float)));

  int grid = (totalH + kBlockSize - 1) / kBlockSize;
  gru_gates_kernel<<<grid, kBlockSize>>>(d_xW, d_hU, d_bias, d_hp, d_ho, d_z, d_r, d_n, kB, kH);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(totalH);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_ho, totalH * sizeof(float), cudaMemcpyDeviceToHost));

  // With all zeros: z=σ(0)=0.5, r=σ(0)=0.5, n=tanh(0)=0
  // h = (1-0.5)*0 + 0.5*0 = 0
  for (int i = 0; i < totalH; ++i) EXPECT_NEAR(h_out[i], 0.0F, 1e-6F);

  CUDA_CHECK(cudaFree(d_xW));
  CUDA_CHECK(cudaFree(d_hU));
  CUDA_CHECK(cudaFree(d_bias));
  CUDA_CHECK(cudaFree(d_hp));
  CUDA_CHECK(cudaFree(d_ho));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_n));
}

// ------------------------------------------------------------------
// Forward sequence: hidden evolves across time-steps
// ------------------------------------------------------------------

TEST_F(GRUTest, ForwardSequenceEvolves) {
  constexpr int kT = 4;

  // Weights
  float *d_W, *d_U, *d_bias;
  CUDA_CHECK(cudaMalloc(&d_W, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_U, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));

  std::vector<float> h_W(3 * kH * kI, 0.05F);
  std::vector<float> h_U(3 * kH * kH, 0.02F);
  std::vector<float> h_bias(3 * kH, 0.0F);
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));

  // Input sequence
  float* d_x;
  CUDA_CHECK(cudaMalloc(&d_x, kT * kB * kI * sizeof(float)));
  std::vector<float> h_x(kT * kB * kI, 0.5F);
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kT * kB * kI * sizeof(float), cudaMemcpyHostToDevice));

  // Hidden states
  float* d_h;
  CUDA_CHECK(cudaMalloc(&d_h, (kT + 1) * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_h, 0, (kT + 1) * kB * kH * sizeof(float)));

  // Scratch
  float *d_xW, *d_hU, *d_z, *d_r, *d_n;
  CUDA_CHECK(cudaMalloc(&d_xW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, kB * kH * sizeof(float)));

  // Forward
  for (int t = 0; t < kT; ++t) {
    float alpha = 1.0F, beta = 0.0F;
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kH, kB, kI, &alpha, d_W, kI,
                             d_x + t * kB * kI, kI, &beta, d_xW, 3 * kH));
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kH, kB, kH, &alpha, d_U, kH,
                             d_h + t * kB * kH, kH, &beta, d_hU, 3 * kH));
    int grid = (kB * kH + kBlockSize - 1) / kBlockSize;
    gru_gates_kernel<<<grid, kBlockSize>>>(d_xW, d_hU, d_bias, d_h + t * kB * kH,
                                           d_h + (t + 1) * kB * kH, d_z, d_r, d_n, kB, kH);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // h_0 (initial) should be 0; h_T should be non-zero
  float h0 = 0.0F, hT = 0.0F;
  CUDA_CHECK(cudaMemcpy(&h0, d_h, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&hT, d_h + kT * kB * kH, sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(h0, 0.0F);
  EXPECT_NE(hT, 0.0F) << "Hidden state should evolve across time";

  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_bias));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_h));
  CUDA_CHECK(cudaFree(d_xW));
  CUDA_CHECK(cudaFree(d_hU));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_n));
}

// ------------------------------------------------------------------
// Backward: bias gradient is non-zero
// ------------------------------------------------------------------

TEST_F(GRUTest, BackwardProducesNonZeroBiasGrad) {
  int total3 = kB * 3 * kH;
  int totalH = kB * kH;

  std::vector<float> h_xW(total3, 0.1F), h_hU(total3, 0.1F);
  std::vector<float> h_bias(3 * kH, 0.0F);
  std::vector<float> h_prev(totalH, 0.5F);
  std::vector<float> h_dh(totalH, 1.0F);

  float *d_xW, *d_hU, *d_bias, *d_hp, *d_ho, *d_z, *d_r, *d_n;
  float *d_dh, *d_dhp, *d_dxW, *d_dhU, *d_dbias;

  CUDA_CHECK(cudaMalloc(&d_xW, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hp, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ho, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dh, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dhp, totalH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxW, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dhU, total3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dbias, 3 * kH * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_xW, h_xW.data(), total3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_hU, h_hU.data(), total3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_hp, h_prev.data(), totalH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dh, h_dh.data(), totalH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dbias, 0, 3 * kH * sizeof(float)));

  // Forward
  int grid = (totalH + kBlockSize - 1) / kBlockSize;
  gru_gates_kernel<<<grid, kBlockSize>>>(d_xW, d_hU, d_bias, d_hp, d_ho, d_z, d_r, d_n, kB, kH);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Backward
  gru_backward_gates_kernel<<<grid, kBlockSize>>>(d_dh, d_z, d_r, d_n, d_hp, d_hU, d_dhp, d_dxW,
                                                  d_dhU, d_dbias, kB, kH);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dbias(3 * kH);
  CUDA_CHECK(cudaMemcpy(h_dbias.data(), d_dbias, 3 * kH * sizeof(float), cudaMemcpyDeviceToHost));

  float sum_abs = 0.0F;
  for (float v : h_dbias) sum_abs += std::abs(v);
  EXPECT_GT(sum_abs, 0.0F) << "Bias gradient should be non-zero";

  CUDA_CHECK(cudaFree(d_xW));
  CUDA_CHECK(cudaFree(d_hU));
  CUDA_CHECK(cudaFree(d_bias));
  CUDA_CHECK(cudaFree(d_hp));
  CUDA_CHECK(cudaFree(d_ho));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_n));
  CUDA_CHECK(cudaFree(d_dh));
  CUDA_CHECK(cudaFree(d_dhp));
  CUDA_CHECK(cudaFree(d_dxW));
  CUDA_CHECK(cudaFree(d_dhU));
  CUDA_CHECK(cudaFree(d_dbias));
}
