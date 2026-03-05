/**
 * @file gru.cu
 * @brief Lesson 30 — Gated Recurrent Unit (GRU).
 *
 * The GRU (Cho et al., 2014) is a lightweight recurrent cell with only **two
 * gates** (update gate z, reset gate r), making it simpler and often faster
 * than LSTM while retaining similar representational power.
 *
 * ## GRU Equations
 *
 * For each time-step t with input x_t and previous hidden state h_{t-1}:
 *
 *     z_t = σ(W_z · x_t + U_z · h_{t-1} + b_z)          (update gate)
 *     r_t = σ(W_r · x_t + U_r · h_{t-1} + b_r)          (reset gate)
 *     ñ_t = tanh(W_n · x_t + U_n · (r_t ⊙ h_{t-1}) + b_n)  (candidate)
 *     h_t = (1 - z_t) ⊙ ñ_t + z_t ⊙ h_{t-1}            (new hidden)
 *
 * where σ is the sigmoid function and ⊙ is element-wise multiplication.
 *
 * ## Implementation Strategy
 *
 * 1. **cuBLAS GEMM** for the weight multiplications (Lesson 19, 28).
 *    We concatenate W_z, W_r, W_n into a single (3H × I) matrix and
 *    similarly for U_z, U_r, U_n → (3H × H) to do one big GEMM each.
 * 2. **Element-wise gate kernel** applies sigmoid / tanh and computes h_t.
 * 3. **BPTT** (Back-Propagation Through Time) unrolls the sequence in
 *    reverse, accumulating gradients.
 *
 * ## Parts
 *
 * - Part 1: Forward — single-step GRU cell
 * - Part 2: Forward — unrolled over T time-steps
 * - Part 3: Backward — BPTT through T steps
 *
 * See Lesson 28 for embeddings that feed into the GRU, and Lesson 25 for
 * optimizers used to train the weights.
 */

#include <cublas_v2.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                          \
  do {                                                                              \
    cublasStatus_t st_ = (call);                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                             \
      std::fprintf(stderr, "cuBLAS error at %s:%d — code %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                          \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Device helpers
// =============================================================================

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

// =============================================================================
// Part 1 — GRU gate kernel (single time-step)
// =============================================================================

/**
 * @brief Apply GRU gates element-wise after GEMM.
 *
 * Inputs xW and hU are pre-computed GEMMs of shape (B × 3H) each.
 * The layout within those 3H columns is [z | r | n].
 *
 * @param xW       (B × 3H) = [W_z·x | W_r·x | W_n·x]
 * @param hU       (B × 3H) = [U_z·h | U_r·h | U_n·h]
 * @param bias     (3H,) concatenated bias [b_z | b_r | b_n]
 * @param h_prev   Previous hidden state (B × H)
 * @param h_out    New hidden state (B × H) — output
 * @param z_buf    (B × H) — stored z for backward
 * @param r_buf    (B × H) — stored r for backward
 * @param n_buf    (B × H) — stored ñ for backward
 * @param B        Batch size
 * @param H        Hidden size
 */
__global__ void gru_gates_kernel(const float* __restrict__ xW, const float* __restrict__ hU,
                                 const float* __restrict__ bias, const float* __restrict__ h_prev,
                                 float* __restrict__ h_out, float* __restrict__ z_buf,
                                 float* __restrict__ r_buf, float* __restrict__ n_buf, int B,
                                 int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * H) return;

  int b = idx / H;
  int h = idx % H;

  // Gather from the concatenated (B × 3H) buffers
  float z_pre = xW[b * 3 * H + h] + hU[b * 3 * H + h] + bias[h];
  float r_pre = xW[b * 3 * H + H + h] + hU[b * 3 * H + H + h] + bias[H + h];

  float z = sigmoid(z_pre);
  float r = sigmoid(r_pre);

  // Candidate: reset gate applied to U_n * h_prev part only
  float n_pre = xW[b * 3 * H + 2 * H + h] + r * hU[b * 3 * H + 2 * H + h] + bias[2 * H + h];
  float n = tanhf(n_pre);

  float hp = h_prev[idx];
  h_out[idx] = (1.0F - z) * n + z * hp;

  z_buf[idx] = z;
  r_buf[idx] = r;
  n_buf[idx] = n;
}

// =============================================================================
// Part 3 — GRU backward gate kernel (single time-step)
// =============================================================================

/**
 * @brief Backward through the GRU gates for a single time-step.
 *
 * Given dh (gradient w.r.t. h_t), computes:
 *   - dh_prev contribution
 *   - d_xW and d_hU (gradients w.r.t. the GEMM outputs)
 *   - dbias contribution
 *
 * @param dh       Gradient w.r.t. h_t (B × H)
 * @param z_buf    Saved z from forward (B × H)
 * @param r_buf    Saved r from forward (B × H)
 * @param n_buf    Saved ñ from forward (B × H)
 * @param h_prev   Previous hidden state (B × H)
 * @param hU       (B × 3H) — the U-based GEMM output from forward
 * @param dh_prev  Output: gradient flowing back to h_{t-1} (B × H)
 * @param d_xW     Output: gradient w.r.t. xW (B × 3H)
 * @param d_hU     Output: gradient w.r.t. hU (B × 3H)
 * @param dbias    Output: gradient w.r.t. bias (3H) — atomicAdd
 * @param B        Batch size
 * @param H        Hidden size
 */
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

  // h_t = (1-z)*n + z*h_{t-1}
  float dn = dh_val * (1.0F - z);
  float dz = dh_val * (hp - n);
  float dh_prev_val = dh_val * z;

  // tanh backward: dn_pre = dn * (1 - n^2)
  float dn_pre = dn * (1.0F - n * n);

  // sigmoid backward: dz_pre = dz * z * (1-z)
  float dz_pre = dz * z * (1.0F - z);

  // n_pre = xW_n + r * hU_n + b_n
  float hU_n = hU[b * 3 * H + 2 * H + h];
  float dr = dn_pre * hU_n;
  float dr_pre = dr * r * (1.0F - r);

  // dh_prev via U_n path: dn_pre * r feeds back through U_n
  // (full gradient through U GEMM is handled by cuBLAS; here we just
  //  record the GEMM-output gradients)
  dh_prev[idx] = dh_prev_val;

  // d_xW: [dz_pre | dr_pre | dn_pre]
  d_xW[b * 3 * H + h] = dz_pre;
  d_xW[b * 3 * H + H + h] = dr_pre;
  d_xW[b * 3 * H + 2 * H + h] = dn_pre;

  // d_hU: [dz_pre | dr_pre | dn_pre * r]
  d_hU[b * 3 * H + h] = dz_pre;
  d_hU[b * 3 * H + H + h] = dr_pre;
  d_hU[b * 3 * H + 2 * H + h] = dn_pre * r;

  // dbias via atomicAdd
  atomicAdd(&dbias[h], dz_pre);
  atomicAdd(&dbias[H + h], dr_pre);
  atomicAdd(&dbias[2 * H + h], dn_pre);
}

// =============================================================================
// GRUCell struct — forward & backward for T steps
// =============================================================================

struct GRUCell {
  int B, I, H;  // Batch, Input dim, Hidden dim
  cublasHandle_t handle;

  // Device weight pointers (owned externally)
  float* W;     // (3H × I)
  float* U;     // (3H × H)
  float* bias;  // (3H)

  /// Forward for a single step: x (B×I), h_prev (B×H) → h_out (B×H)
  void forward_step(const float* d_x, const float* d_h_prev, float* d_h_out, float* xW_buf,
                    float* hU_buf, float* z_buf, float* r_buf, float* n_buf) {
    float alpha = 1.0F, beta = 0.0F;

    // xW = x @ W^T  →  (B × 3H)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * H, B, I, &alpha, W, I, d_x, I,
                             &beta, xW_buf, 3 * H));

    // hU = h_prev @ U^T  →  (B × 3H)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * H, B, H, &alpha, U, H, d_h_prev,
                             H, &beta, hU_buf, 3 * H));

    int grid = (B * H + kBlockSize - 1) / kBlockSize;
    gru_gates_kernel<<<grid, kBlockSize>>>(xW_buf, hU_buf, bias, d_h_prev, d_h_out, z_buf, r_buf,
                                           n_buf, B, H);
    CUDA_CHECK(cudaGetLastError());
  }
};

// =============================================================================
// main
// =============================================================================

int main() {
  constexpr int kB = 4;
  constexpr int kI = 32;
  constexpr int kH = 64;
  constexpr int kT = 8;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---- Allocate weights ----
  float* d_W;
  float* d_U;
  float* d_bias;
  CUDA_CHECK(cudaMalloc(&d_W, 3 * kH * kI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_U, 3 * kH * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bias, 3 * kH * sizeof(float)));

  // Simple init
  std::vector<float> h_W(3 * kH * kI, 0.01F);
  std::vector<float> h_U(3 * kH * kH, 0.01F);
  std::vector<float> h_bias(3 * kH, 0.0F);
  CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), 3 * kH * kI * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), 3 * kH * kH * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 3 * kH * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Allocate sequence IO ----
  float* d_x;  // (T × B × I)
  CUDA_CHECK(cudaMalloc(&d_x, kT * kB * kI * sizeof(float)));
  std::vector<float> h_x(kT * kB * kI);
  for (int i = 0; i < static_cast<int>(h_x.size()); ++i)
    h_x[i] = static_cast<float>(i % 11) * 0.02F - 0.1F;
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kT * kB * kI * sizeof(float), cudaMemcpyHostToDevice));

  // Hidden states: T+1 snapshots (h_0 = 0)
  float* d_h;  // ((T+1) × B × H)
  CUDA_CHECK(cudaMalloc(&d_h, (kT + 1) * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_h, 0, (kT + 1) * kB * kH * sizeof(float)));

  // Scratch for GEMM outputs and gate saves
  float* d_xW;
  float* d_hU;
  float* d_z;
  float* d_r;
  float* d_n;
  CUDA_CHECK(cudaMalloc(&d_xW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, kT * kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_n, kT * kB * kH * sizeof(float)));

  GRUCell cell{kB, kI, kH, handle, d_W, d_U, d_bias};

  // ---- Part 2: Forward T steps ----
  std::printf("=== GRU Forward (%d steps) ===\n", kT);
  for (int t = 0; t < kT; ++t) {
    const float* x_t = d_x + t * kB * kI;
    const float* h_prev = d_h + t * kB * kH;
    float* h_next = d_h + (t + 1) * kB * kH;
    float* z_t = d_z + t * kB * kH;
    float* r_t = d_r + t * kB * kH;
    float* n_t = d_n + t * kB * kH;

    cell.forward_step(x_t, h_prev, h_next, d_xW, d_hU, z_t, r_t, n_t);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Print final hidden state snippet
  float h_final = 0.0F;
  CUDA_CHECK(cudaMemcpy(&h_final, d_h + kT * kB * kH, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  h[T][0][0] = %.6f\n", static_cast<double>(h_final));

  // ---- Part 3: Backward (BPTT) ----
  std::printf("=== GRU Backward (BPTT) ===\n");

  float* d_dh;  // gradient w.r.t. hidden state at each step
  float* d_dh_prev;
  float* d_d_xW;
  float* d_d_hU;
  float* d_dbias;
  CUDA_CHECK(cudaMalloc(&d_dh, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dh_prev, kB * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_d_xW, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_d_hU, kB * 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dbias, 3 * kH * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dbias, 0, 3 * kH * sizeof(float)));

  // Seed: gradient of loss w.r.t. final hidden state = 1.0
  std::vector<float> ones(kB * kH, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dh, ones.data(), kB * kH * sizeof(float), cudaMemcpyHostToDevice));

  for (int t = kT - 1; t >= 0; --t) {
    const float* h_prev = d_h + t * kB * kH;
    const float* z_t = d_z + t * kB * kH;
    const float* r_t = d_r + t * kB * kH;
    const float* n_t = d_n + t * kB * kH;

    // Re-compute hU for this step (same GEMM as forward)
    float alpha = 1.0F, beta = 0.0F;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * kH, kB, kH, &alpha, d_U, kH,
                             h_prev, kH, &beta, d_hU, 3 * kH));

    int grid = (kB * kH + kBlockSize - 1) / kBlockSize;
    gru_backward_gates_kernel<<<grid, kBlockSize>>>(d_dh, z_t, r_t, n_t, h_prev, d_hU, d_dh_prev,
                                                    d_d_xW, d_d_hU, d_dbias, kB, kH);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Propagate: dh for next iteration (t-1) includes dh_prev + gradient through U
    //
    // NOTE: Pedagogical simplification — a complete BPTT implementation would
    // add the gradient flowing through the hidden-to-hidden (U) matrix:
    //     dh_{t-1} += d_gates @ U^T
    // and would also accumulate weight gradients dW and dU via outer products.
    // Here we only propagate dh_prev directly and accumulate bias gradients to
    // keep the lesson focused on the gate-level backward computation.
    CUDA_CHECK(cudaMemcpy(d_dh, d_dh_prev, kB * kH * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // Print bias gradient snippet
  float dbias0 = 0.0F;
  CUDA_CHECK(cudaMemcpy(&dbias0, d_dbias, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  dbias[0] = %.6f\n", static_cast<double>(dbias0));

  // ---- Cleanup ----
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
  CUDA_CHECK(cudaFree(d_d_xW));
  CUDA_CHECK(cudaFree(d_d_hU));
  CUDA_CHECK(cudaFree(d_dbias));
  CUBLAS_CHECK(cublasDestroy(handle));

  std::printf("\nDone.\n");
  return EXIT_SUCCESS;
}
