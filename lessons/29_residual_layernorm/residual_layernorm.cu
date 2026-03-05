/**
 * @file residual_layernorm.cu
 * @brief Lesson 29 — Residual Connections & Layer Normalization.
 *
 * Residual connections and layer normalization are the two structural
 * primitives that make deep Transformer models trainable.  This lesson
 * implements both — plus a **fused** kernel that performs the add + norm in
 * one pass — and their backward passes.
 *
 * ## Part 1 — Residual Connection
 *
 * A residual (skip) connection simply adds the block's input to its
 * output:
 *
 *     y = x + F(x)
 *
 * where F is any sub‑layer (attention, FFN, etc.).  The backward pass is
 * trivial: gradients flow through both branches unmodified.
 *
 * ## Part 2 — Layer Normalization (Forward)
 *
 * Unlike batch normalization (Lesson 23), layer norm normalizes across the
 * **feature dimension** of each individual sample:
 *
 *     mu  = (1/D) * Σ x_d
 *     var = (1/D) * Σ (x_d - mu)^2
 *     x_hat = (x - mu) / sqrt(var + eps)
 *     y = gamma * x_hat + beta
 *
 * We use **Welford's online algorithm** inside a single warp reduction for
 * numerical stability.
 *
 * ## Part 3 — Layer Normalization (Backward)
 *
 * The backward pass computes dx, dgamma, dbeta given dy:
 *
 *     dx_hat = dy * gamma
 *     dvar   = sum(dx_hat * (x - mu)) * (-0.5) * (var + eps)^{-3/2}
 *     dmu    = sum(dx_hat) * (-1/sqrt(var+eps)) + dvar * (-2/D) * sum(x-mu)
 *     dx     = dx_hat / sqrt(var+eps) + dvar * 2*(x-mu)/D + dmu/D
 *
 * ## Part 4 — Fused Residual + LayerNorm
 *
 * In practice the residual add and layer norm are fused into a single
 * kernel to save a global memory round‑trip.
 *
 * See Lesson 23 for batch normalization, and Lesson 32 for how residual +
 * layer norm slots into the Transformer encoder block.
 */

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

constexpr int kBlockSize = 256;

// =============================================================================
// Part 1 — Residual Connection
// =============================================================================

/**
 * @brief y[i] = x[i] + residual[i].
 *
 * Forward pass of a residual (skip) connection.
 *
 * @param x         Input tensor.
 * @param residual  Residual (skip) tensor.
 * @param y         Output tensor.
 * @param n         Number of elements.
 */
__global__ void residual_add_kernel(const float* __restrict__ x, const float* __restrict__ residual,
                                    float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + residual[i];
}

/**
 * @brief Backward: dx[i] = dy[i], dresidual[i] = dy[i].
 *
 * Gradients flow identically to both branches.
 *
 * @param dy         Upstream gradient.
 * @param dx         Output gradient w.r.t. x.
 * @param dresidual  Output gradient w.r.t. residual.
 * @param n          Number of elements.
 */
__global__ void residual_backward_kernel(const float* __restrict__ dy, float* __restrict__ dx,
                                         float* __restrict__ dresidual, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dx[i] = dy[i];
    dresidual[i] = dy[i];
  }
}

// =============================================================================
// Part 2 — Layer Normalization Forward
// =============================================================================

/**
 * @brief One‑block‑per‑row layer norm using Welford's online algorithm.
 *
 * Each block normalizes one row (D elements).  We assume D ≤ blockDim.x
 * for simplicity; a production implementation would tile.
 *
 * @param x      Input  (N × D), row‑major
 * @param y      Output (N × D)
 * @param gamma  Scale  (D,)
 * @param beta   Shift  (D,)
 * @param mean   Per‑row mean output (N,)   — stored for backward
 * @param rstd   Per‑row 1/sqrt(var+eps) (N,) — stored for backward
 * @param N      Number of rows
 * @param D      Feature dimension
 * @param eps    Epsilon for numerical stability
 */
__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ mean,
                                         float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;

  const float* row_in = x + row * D;

  // ---- Welford mean ----
  extern __shared__ float smem[];  // [0..blockDim.x) partial sums
  float local_sum = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) local_sum += row_in[d];
  smem[threadIdx.x] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);

  // ---- Variance ----
  float local_var = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float diff = row_in[d] - mu;
    local_var += diff * diff;
  }
  smem[threadIdx.x] = local_var;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float var = smem[0] / static_cast<float>(D);
  float inv_std = rsqrtf(var + eps);

  // ---- Normalize ----
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }

  float* row_out = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (row_in[d] - mu) * inv_std;
    row_out[d] = gamma[d] * x_hat + beta[d];
  }
}

// =============================================================================
// Part 3 — Layer Normalization Backward
// =============================================================================

/**
 * @brief Layer norm backward — dx, dgamma, dbeta.
 *
 * Each block handles one row for dx.  dgamma and dbeta are accumulated with
 * atomicAdd across rows.
 *
 * @param dy      Upstream gradient (N × D).
 * @param x       Input from forward pass (N × D).
 * @param gamma   Scale parameter (D,).
 * @param mean    Per-row mean from forward pass (N,).
 * @param rstd    Per-row 1/sqrt(var+eps) from forward pass (N,).
 * @param dx      Output gradient w.r.t. input (N × D).
 * @param dgamma  Output gradient w.r.t. gamma (D,), atomically accumulated.
 * @param dbeta   Output gradient w.r.t. beta (D,), atomically accumulated.
 * @param N       Number of rows.
 * @param D       Feature dimension.
 */
__global__ void layernorm_backward_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ rstd, float* __restrict__ dx,
                                          float* __restrict__ dgamma, float* __restrict__ dbeta,
                                          int N, int D) {
  int row = blockIdx.x;
  if (row >= N) return;

  const float* dy_row = dy + row * D;
  const float* x_row = x + row * D;
  float mu = mean[row];
  float inv_std = rstd[row];

  extern __shared__ float smem[];
  // smem[0..blockDim.x): partial dot(dy*gamma, x_hat)
  // smem[blockDim.x..2*blockDim.x): partial sum(dy*gamma)

  float dot1 = 0.0F;
  float dot2 = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] - mu) * inv_std;
    float dxhat = dy_row[d] * gamma[d];
    dot1 += dxhat * x_hat;
    dot2 += dxhat;
  }
  smem[threadIdx.x] = dot1;
  smem[threadIdx.x + blockDim.x] = dot2;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
      smem[threadIdx.x + blockDim.x] += smem[threadIdx.x + blockDim.x + s];
    }
    __syncthreads();
  }
  float sum_dxhat_xhat = smem[0];
  float sum_dxhat = smem[blockDim.x];

  float inv_D = 1.0F / static_cast<float>(D);

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] - mu) * inv_std;
    float dxhat = dy_row[d] * gamma[d];
    dx[row * D + d] = inv_std * (dxhat - inv_D * (sum_dxhat + x_hat * sum_dxhat_xhat));

    // Accumulate dgamma, dbeta across rows
    atomicAdd(&dgamma[d], dy_row[d] * x_hat);
    atomicAdd(&dbeta[d], dy_row[d]);
  }
}

// =============================================================================
// Part 4 — Fused Residual + LayerNorm
// =============================================================================

/**
 * @brief y = LayerNorm(x + residual)  in a single kernel.
 *
 * Saves one global‑memory read/write compared to separate add + norm.
 *
 * @param x         Input tensor (N × D).
 * @param residual  Residual tensor (N × D).
 * @param y         Output tensor (N × D).
 * @param gamma     Scale parameter (D,).
 * @param beta      Shift parameter (D,).
 * @param mean      Per-row mean output (N,).
 * @param rstd      Per-row 1/sqrt(var+eps) output (N,).
 * @param N         Number of rows.
 * @param D         Feature dimension.
 * @param eps       Epsilon for numerical stability.
 */
__global__ void fused_residual_layernorm_kernel(
    const float* __restrict__ x, const float* __restrict__ residual, float* __restrict__ y,
    const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ mean,
    float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;

  const float* x_row = x + row * D;
  const float* r_row = residual + row * D;

  extern __shared__ float smem[];

  // Sum for mean
  float local_sum = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    local_sum += x_row[d] + r_row[d];
  }
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);

  // Variance
  float local_var = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float val = x_row[d] + r_row[d] - mu;
    local_var += val * val;
  }
  smem[threadIdx.x] = local_var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);

  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }

  float* out_row = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] + r_row[d] - mu) * inv_std;
    out_row[d] = gamma[d] * x_hat + beta[d];
  }
}

// =============================================================================
// main
// =============================================================================

int main() {
  constexpr int kN = 128;
  constexpr int kD = 256;
  constexpr float kEps = 1e-5F;

  // ---- Host data ----
  std::vector<float> h_x(kN * kD), h_residual(kN * kD);
  std::vector<float> h_gamma(kD, 1.0F), h_beta(kD, 0.0F);
  for (int i = 0; i < kN * kD; ++i) {
    h_x[i] = static_cast<float>(i % 17) * 0.1F - 0.8F;
    h_residual[i] = static_cast<float>(i % 13) * 0.05F - 0.3F;
  }

  // ---- Device ----
  float *d_x, *d_res, *d_y, *d_gamma, *d_beta, *d_mean, *d_rstd;
  CUDA_CHECK(cudaMalloc(&d_x, kN * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_res, kN * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, kN * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mean, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rstd, kN * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kN * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_res, h_residual.data(), kN * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), kD * sizeof(float), cudaMemcpyHostToDevice));

  // Part 1: Residual add
  std::printf("=== Part 1: Residual Add ===\n");
  {
    int grid = (kN * kD + kBlockSize - 1) / kBlockSize;
    residual_add_kernel<<<grid, kBlockSize>>>(d_x, d_res, d_y, kN * kD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float v = 0.0F;
    CUDA_CHECK(cudaMemcpy(&v, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  y[0] = %.4f (x=%.4f + res=%.4f)\n", static_cast<double>(v),
                static_cast<double>(h_x[0]), static_cast<double>(h_residual[0]));
  }

  // Part 2: LayerNorm forward
  std::printf("=== Part 2: Layer Normalization ===\n");
  {
    int block = (kD < kBlockSize) ? kD : kBlockSize;
    // Ensure block is power of 2
    int b2 = 1;
    while (b2 < block) b2 <<= 1;
    block = b2;
    size_t smem = block * sizeof(float);
    layernorm_forward_kernel<<<kN, block, smem>>>(d_x, d_y, d_gamma, d_beta, d_mean, d_rstd, kN, kD,
                                                  kEps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float m = 0.0F, r = 0.0F;
    CUDA_CHECK(cudaMemcpy(&m, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&r, d_rstd, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  Row 0: mean=%.6f  rstd=%.6f\n", static_cast<double>(m), static_cast<double>(r));
  }

  // Part 4: Fused residual + layernorm
  std::printf("=== Part 4: Fused Residual + LayerNorm ===\n");
  {
    int block = (kD < kBlockSize) ? kD : kBlockSize;
    int b2 = 1;
    while (b2 < block) b2 <<= 1;
    block = b2;
    size_t smem = block * sizeof(float);
    fused_residual_layernorm_kernel<<<kN, block, smem>>>(d_x, d_res, d_y, d_gamma, d_beta, d_mean,
                                                         d_rstd, kN, kD, kEps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float m = 0.0F;
    CUDA_CHECK(cudaMemcpy(&m, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  Row 0 fused mean = %.6f\n", static_cast<double>(m));
  }

  // Part 3: LayerNorm backward
  std::printf("=== Part 3: LayerNorm Backward ===\n");
  {
    // Allocate upstream gradient (dy), output (dx), dgamma, dbeta
    float *d_dy, *d_dx, *d_dgamma, *d_dbeta;
    CUDA_CHECK(cudaMalloc(&d_dy, kN * kD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, kN * kD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dgamma, kD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbeta, kD * sizeof(float)));

    // Seed upstream gradient with ones
    std::vector<float> h_dy(kN * kD, 1.0F);
    CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), kN * kD * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dgamma, 0, kD * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbeta, 0, kD * sizeof(float)));

    // Re-run forward so mean/rstd are valid for the raw input d_x
    int block = (kD < kBlockSize) ? kD : kBlockSize;
    int b2 = 1;
    while (b2 < block) b2 <<= 1;
    block = b2;
    size_t smem = 2 * block * sizeof(float);  // backward needs 2× smem

    layernorm_forward_kernel<<<kN, block, block * sizeof(float)>>>(d_x, d_y, d_gamma, d_beta,
                                                                   d_mean, d_rstd, kN, kD, kEps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    layernorm_backward_kernel<<<kN, block, smem>>>(d_dy, d_x, d_gamma, d_mean, d_rstd, d_dx,
                                                   d_dgamma, d_dbeta, kN, kD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float dx0 = 0.0F;
    CUDA_CHECK(cudaMemcpy(&dx0, d_dx, sizeof(float), cudaMemcpyDeviceToHost));
    float dg0 = 0.0F;
    CUDA_CHECK(cudaMemcpy(&dg0, d_dgamma, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("  dx[0] = %.6f  dgamma[0] = %.6f\n", static_cast<double>(dx0),
                static_cast<double>(dg0));

    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dgamma));
    CUDA_CHECK(cudaFree(d_dbeta));
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_res));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_rstd));

  std::printf("\nDone.\n");
  return EXIT_SUCCESS;
}
