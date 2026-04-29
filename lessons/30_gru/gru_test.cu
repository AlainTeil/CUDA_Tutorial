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
    const cudaError_t err_ = (call);                                        \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    const cublasStatus_t st_ = (call);                                        \
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

// ------------------------------------------------------------------
// Full BPTT: weight gradients match finite differences
// ------------------------------------------------------------------

/// Run a full forward pass over T steps, return sum of final hidden state
/// (a scalar loss).  All buffers are caller-owned.
static float run_forward_and_loss(cublasHandle_t handle, const float* d_W, const float* d_U,
                                  const float* d_bias, const float* d_x, float* d_h, float* d_xW,
                                  float* d_hU, float* d_z, float* d_r, float* d_n, int B, int I,
                                  int H, int T) {
  // h_0 = 0
  CUDA_CHECK(cudaMemset(d_h, 0, (T + 1) * B * H * sizeof(float)));

  float alpha = 1.0F, beta = 0.0F;
  for (int t = 0; t < T; ++t) {
    const float* x_t = d_x + t * B * I;
    const float* h_prev = d_h + t * B * H;
    float* h_next = d_h + (t + 1) * B * H;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * H, B, I, &alpha, d_W, I, x_t, I,
                             &beta, d_xW, 3 * H));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * H, B, H, &alpha, d_U, H, h_prev,
                             H, &beta, d_hU, 3 * H));
    int grid = (B * H + kBlockSize - 1) / kBlockSize;
    gru_gates_kernel<<<grid, kBlockSize>>>(d_xW, d_hU, d_bias, h_prev, h_next, d_z + t * B * H,
                                           d_r + t * B * H, d_n + t * B * H, B, H);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Loss = sum(h_T)
  std::vector<float> h_final(B * H);
  CUDA_CHECK(
      cudaMemcpy(h_final.data(), d_h + T * B * H, B * H * sizeof(float), cudaMemcpyDeviceToHost));
  float loss = 0.0F;
  for (float v : h_final) loss += v;
  return loss;
}

TEST_F(GRUTest, BPTTWeightGradsMatchFiniteDiff) {
  constexpr int kT = 3;
  constexpr float kEps = 5e-3F;  // finite-diff step (larger for float32 stability)
  constexpr float kTol = 5e-2F;  // 5% relative tolerance
  // Floor for the denominator: gradients smaller than this are compared via
  // absolute error (kAbsFloor * kTol) instead of relative error.
  constexpr float kAbsFloor = 1e-3F;

  // ---- Allocate ----
  float *d_W, *d_U, *d_bias, *d_x, *d_h;
  float *d_xW, *d_hU, *d_z, *d_r, *d_n;
  CUDA_CHECK(cudaMalloc(&d_W, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_U, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, kT * kB * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_h, (kT + 1) * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, kT * kB * kH * sizeof(float)));

  // Larger weights produce larger gradients → more reliable finite diffs
  std::vector<float> h_W(3 * kH * kI), h_U(3 * kH * kH), h_bias(3 * kH);
  std::vector<float> h_x(kT * kB * kI);
  for (int i = 0; i < static_cast<int>(h_W.size()); ++i)
    h_W[i] = static_cast<float>((i % 13) - 6) * 0.08F;
  for (int i = 0; i < static_cast<int>(h_U.size()); ++i)
    h_U[i] = static_cast<float>((i % 11) - 5) * 0.06F;
  for (int i = 0; i < 3 * kH; ++i) h_bias[i] = static_cast<float>((i % 7) - 3) * 0.05F;
  for (int i = 0; i < static_cast<int>(h_x.size()); ++i)
    h_x[i] = static_cast<float>((i % 9) - 4) * 0.1F;

  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kT * kB * kI * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Analytical gradients via full BPTT ----

  // Forward (saves gates)
  run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z, d_r, d_n, kB, kI, kH,
                       kT);

  // Backward
  float *d_dh, *d_dh_prev, *d_dxW, *d_dhU, *d_dbias, *d_dW, *d_dU;
  CUDA_CHECK(cudaMalloc(&d_dh, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dh_prev, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dhU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dbias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dW, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dU, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dbias, 0, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dW, 0, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dU, 0, 3 * kH * kH * sizeof(float)));

  // Seed: dL/dh_T = 1 (since loss = sum(h_T))
  std::vector<float> ones(kB * kH, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dh, ones.data(), kB * kH * sizeof(float), cudaMemcpyHostToDevice));

  float alpha_blas = 1.0F, beta0 = 0.0F, beta1 = 1.0F;
  for (int t = kT - 1; t >= 0; --t) {
    const float* x_t = d_x + t * kB * kI;
    const float* h_prev = d_h + t * kB * kH;

    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kH, kB, kH, &alpha_blas, d_U,
                             kH, h_prev, kH, &beta0, d_hU, 3 * kH));
    int grid = (kB * kH + kBlockSize - 1) / kBlockSize;
    gru_backward_gates_kernel<<<grid, kBlockSize>>>(d_dh, d_z + t * kB * kH, d_r + t * kB * kH,
                                                    d_n + t * kB * kH, h_prev, d_hU, d_dh_prev,
                                                    d_dxW, d_dhU, d_dbias, kB, kH);
    CUDA_CHECK(cudaGetLastError());

    // dW += x_t^T @ d_xW
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, kI, 3 * kH, kB, &alpha_blas, x_t,
                             kI, d_dxW, 3 * kH, &beta1, d_dW, kI));
    // dU += h_prev^T @ d_hU
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, kH, 3 * kH, kB, &alpha_blas, h_prev,
                             kH, d_dhU, 3 * kH, &beta1, d_dU, kH));
    // dh_prev += U @ d_hU
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kH, kB, 3 * kH, &alpha_blas, d_U,
                             kH, d_dhU, 3 * kH, &beta1, d_dh_prev, kH));

    CUDA_CHECK(cudaMemcpy(d_dh, d_dh_prev, kB * kH * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read analytical gradients
  std::vector<float> ana_dW(3 * kH * kI), ana_dU(3 * kH * kH), ana_dbias(3 * kH);
  CUDA_CHECK(cudaMemcpy(ana_dW.data(), d_dW, 3 * kH * kI * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ana_dU.data(), d_dU, 3 * kH * kH * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ana_dbias.data(), d_dbias, 3 * kH * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- Numerical gradients (central finite differences) ----
  // Check a subset of W elements to keep the test fast
  constexpr int kCheckW = 16;
  for (int idx = 0; idx < kCheckW; ++idx) {
    float orig = h_W[idx];

    h_W[idx] = orig + kEps;
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
    float loss_plus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                           d_r, d_n, kB, kI, kH, kT);

    h_W[idx] = orig - kEps;
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
    float loss_minus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                            d_r, d_n, kB, kI, kH, kT);

    h_W[idx] = orig;
    float num_grad = (loss_plus - loss_minus) / (2.0F * kEps);
    float ana_grad = ana_dW[idx];
    float denom = std::max({std::abs(num_grad), std::abs(ana_grad), kAbsFloor});
    float rel_err = std::abs(num_grad - ana_grad) / denom;
    EXPECT_LT(rel_err, kTol) << "dW[" << idx << "] analytical=" << ana_grad
                             << " numerical=" << num_grad;
  }

  // Restore W
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));

  // Check a subset of U elements
  constexpr int kCheckU = 16;
  for (int idx = 0; idx < kCheckU; ++idx) {
    float orig = h_U[idx];

    h_U[idx] = orig + kEps;
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
    float loss_plus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                           d_r, d_n, kB, kI, kH, kT);

    h_U[idx] = orig - kEps;
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
    float loss_minus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                            d_r, d_n, kB, kI, kH, kT);

    h_U[idx] = orig;
    float num_grad = (loss_plus - loss_minus) / (2.0F * kEps);
    float ana_grad = ana_dU[idx];
    float denom = std::max({std::abs(num_grad), std::abs(ana_grad), kAbsFloor});
    float rel_err = std::abs(num_grad - ana_grad) / denom;
    EXPECT_LT(rel_err, kTol) << "dU[" << idx << "] analytical=" << ana_grad
                             << " numerical=" << num_grad;
  }

  // Restore U
  CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));

  // Check a subset of bias elements
  constexpr int kCheckBias = std::min(3 * kH, 12);
  for (int idx = 0; idx < kCheckBias; ++idx) {
    float orig = h_bias[idx];

    h_bias[idx] = orig + kEps;
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
    float loss_plus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                           d_r, d_n, kB, kI, kH, kT);

    h_bias[idx] = orig - kEps;
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
    float loss_minus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                            d_r, d_n, kB, kI, kH, kT);

    h_bias[idx] = orig;
    float num_grad = (loss_plus - loss_minus) / (2.0F * kEps);
    float ana_grad = ana_dbias[idx];
    float denom = std::max({std::abs(num_grad), std::abs(ana_grad), kAbsFloor});
    float rel_err = std::abs(num_grad - ana_grad) / denom;
    EXPECT_LT(rel_err, kTol) << "dbias[" << idx << "] analytical=" << ana_grad
                             << " numerical=" << num_grad;
  }

  // Cleanup
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
  CUDA_CHECK(cudaFree(d_dh));
  CUDA_CHECK(cudaFree(d_dh_prev));
  CUDA_CHECK(cudaFree(d_dxW));
  CUDA_CHECK(cudaFree(d_dhU));
  CUDA_CHECK(cudaFree(d_dbias));
  CUDA_CHECK(cudaFree(d_dW));
  CUDA_CHECK(cudaFree(d_dU));
}

// ------------------------------------------------------------------
// Full BPTT: input gradients (dx) match finite differences
// ------------------------------------------------------------------
//
// Complements BPTTWeightGradsMatchFiniteDiff by probing the gradient that
// flows back to the inputs (dx_t = W @ d_xW), which is the path used when
// chaining a GRU after an embedding / preceding layer.
TEST_F(GRUTest, BPTTInputGradMatchesFiniteDiff) {
  constexpr int kT = 3;
  constexpr float kEps = 5e-3F;
  constexpr float kTol = 5e-2F;
  constexpr float kAbsFloor = 1e-3F;

  // ---- Allocate ----
  float *d_W, *d_U, *d_bias, *d_x, *d_h;
  float *d_xW, *d_hU, *d_z, *d_r, *d_n;
  CUDA_CHECK(cudaMalloc(&d_W, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_U, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, kT * kB * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_h, (kT + 1) * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, kT * kB * kH * sizeof(float)));

  std::vector<float> h_W(3 * kH * kI), h_U(3 * kH * kH), h_bias(3 * kH);
  std::vector<float> h_x(kT * kB * kI);
  for (int i = 0; i < static_cast<int>(h_W.size()); ++i)
    h_W[i] = static_cast<float>((i % 13) - 6) * 0.08F;
  for (int i = 0; i < static_cast<int>(h_U.size()); ++i)
    h_U[i] = static_cast<float>((i % 11) - 5) * 0.06F;
  for (int i = 0; i < 3 * kH; ++i) h_bias[i] = static_cast<float>((i % 7) - 3) * 0.05F;
  for (int i = 0; i < static_cast<int>(h_x.size()); ++i)
    h_x[i] = static_cast<float>((i % 9) - 4) * 0.1F;

  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kT * kB * kI * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Analytical: full BPTT including dx ----
  run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z, d_r, d_n, kB, kI, kH,
                       kT);

  float *d_dh, *d_dh_prev, *d_dxW, *d_dhU, *d_dbias, *d_dW, *d_dU, *d_dx;
  CUDA_CHECK(cudaMalloc(&d_dh, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dh_prev, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dxW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dhU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dbias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dW, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dU, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, kT * kB * kI * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dbias, 0, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dW, 0, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dU, 0, 3 * kH * kH * sizeof(float)));

  std::vector<float> ones(kB * kH, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dh, ones.data(), kB * kH * sizeof(float), cudaMemcpyHostToDevice));

  float alpha_blas = 1.0F, beta0 = 0.0F, beta1 = 1.0F;
  for (int t = kT - 1; t >= 0; --t) {
    const float* x_t = d_x + t * kB * kI;
    const float* h_prev = d_h + t * kB * kH;

    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kH, kB, kH, &alpha_blas, d_U,
                             kH, h_prev, kH, &beta0, d_hU, 3 * kH));
    int grid = (kB * kH + kBlockSize - 1) / kBlockSize;
    gru_backward_gates_kernel<<<grid, kBlockSize>>>(d_dh, d_z + t * kB * kH, d_r + t * kB * kH,
                                                    d_n + t * kB * kH, h_prev, d_hU, d_dh_prev,
                                                    d_dxW, d_dhU, d_dbias, kB, kH);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, kI, 3 * kH, kB, &alpha_blas, x_t,
                             kI, d_dxW, 3 * kH, &beta1, d_dW, kI));
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, kH, 3 * kH, kB, &alpha_blas, h_prev,
                             kH, d_dhU, 3 * kH, &beta1, d_dU, kH));
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kH, kB, 3 * kH, &alpha_blas, d_U,
                             kH, d_dhU, 3 * kH, &beta1, d_dh_prev, kH));

    // dx_t = W @ d_xW   →  (I × 3H) @ (3H × B) = (I × B)
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kI, kB, 3 * kH, &alpha_blas, d_W,
                             kI, d_dxW, 3 * kH, &beta0, d_dx + t * kB * kI, kI));

    CUDA_CHECK(cudaMemcpy(d_dh, d_dh_prev, kB * kH * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> ana_dx(kT * kB * kI);
  CUDA_CHECK(cudaMemcpy(ana_dx.data(), d_dx, kT * kB * kI * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- Numerical: probe a subset of input elements across all time-steps ----
  // Strided sampling so we cover early, middle, and late steps.
  const int total_x = kT * kB * kI;
  const int kCheck = 18;
  const int stride = total_x / kCheck;
  for (int k = 0; k < kCheck; ++k) {
    int idx = k * stride;
    float orig = h_x[idx];

    h_x[idx] = orig + kEps;
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_x * sizeof(float), cudaMemcpyHostToDevice));
    float loss_plus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                           d_r, d_n, kB, kI, kH, kT);

    h_x[idx] = orig - kEps;
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_x * sizeof(float), cudaMemcpyHostToDevice));
    float loss_minus = run_forward_and_loss(handle_, d_W, d_U, d_bias, d_x, d_h, d_xW, d_hU, d_z,
                                            d_r, d_n, kB, kI, kH, kT);

    h_x[idx] = orig;
    float num_grad = (loss_plus - loss_minus) / (2.0F * kEps);
    float ana_grad = ana_dx[idx];
    float denom = std::max({std::abs(num_grad), std::abs(ana_grad), kAbsFloor});
    float rel_err = std::abs(num_grad - ana_grad) / denom;
    EXPECT_LT(rel_err, kTol) << "dx[" << idx << "] analytical=" << ana_grad
                             << " numerical=" << num_grad;
  }

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
  CUDA_CHECK(cudaFree(d_dh));
  CUDA_CHECK(cudaFree(d_dh_prev));
  CUDA_CHECK(cudaFree(d_dxW));
  CUDA_CHECK(cudaFree(d_dhU));
  CUDA_CHECK(cudaFree(d_dbias));
  CUDA_CHECK(cudaFree(d_dW));
  CUDA_CHECK(cudaFree(d_dU));
  CUDA_CHECK(cudaFree(d_dx));
}
