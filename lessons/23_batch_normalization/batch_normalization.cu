/**
 * @file batch_normalization.cu
 * @brief Lesson 23 — Batch Normalization.
 *
 * **Batch Normalization** (Ioffe & Szegedy, 2015) normalises the activations
 * of each feature across the **mini-batch dimension** so that every feature
 * has approximately zero mean and unit variance before the next layer sees it.
 *
 * ## Why Batch Normalization?
 *
 * During training the distribution of each layer's inputs shifts as the
 * parameters of the preceding layers change — a phenomenon called
 * **internal covariate shift**.  This forces downstream layers to
 * continuously adapt to a moving target, slowing convergence.  Batch
 * Normalization decouples the layers by standardising each feature:
 *
 *     x̂_i = (x_i − μ) / √(σ² + ε)        (1)  normalise
 *     y_i = γ · x̂_i + β                    (2)  scale & shift
 *
 * where μ, σ² are the mean and variance **across the batch** for a single
 * feature, and γ (scale) and β (shift) are learnable parameters that let the
 * network undo the normalisation when that is beneficial.
 *
 * ## Training vs. Inference
 *
 * | Mode      | Statistics used        | Updated?                          |
 * |-----------|------------------------|-----------------------------------|
 * | Training  | Per-batch μ, σ²        | Running mean/var with momentum    |
 * | Inference | Running μ, σ² (fixed)  | No updates                        |
 *
 * During training we also maintain **exponential-moving-average** running
 * statistics (momentum = 0.1 by default) so that at inference time we do
 * not depend on the current mini-batch.
 *
 * ## CUDA implementation
 *
 * The forward pass requires two reductions per feature (mean & variance),
 * each of which follows the shared-memory reduction pattern introduced in
 * Lesson 08.  We assign **one block per feature**, and threads within the
 * block cooperatively iterate over the batch dimension.
 *
 * The backward pass computes three quantities:
 *
 *   - dγ (dloss/dγ) — a reduction of upstream gradient × normalised input
 *   - dβ (dloss/dβ) — a reduction of upstream gradient
 *   - dx (dloss/dx) — per-element, using dγ, dβ, and the batch statistics
 *
 * The dx formula (derived from the chain rule through the normalisation):
 *
 *     dx_i = (1/√(σ²+ε)) · [dŷ_i − (1/N)(dβ + x̂_i · dγ)]
 *
 * See Lesson 16 for the similarly structured loss backward, and Lesson 08
 * for the shared-memory reduction kernel this lesson builds upon.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (call);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Forward — compute per-feature mean
// =============================================================================

/**
 * @brief Compute the per-channel mean over the batch.
 *
 * Grid: one block per feature (gridDim.x = C).  Threads cooperate to sum
 * over N samples.  Input layout: (N, C) row-major.
 *
 * @param x    Input activations (N × C)
 * @param mean Output per-channel means [C]
 * @param N    Batch size
 * @param C    Number of features / channels
 */
__global__ void compute_mean_kernel(const float* __restrict__ x, float* __restrict__ mean, int N,
                                    int C) {
  extern __shared__ float sdata[];
  int c = blockIdx.x;  // feature index
  int tid = threadIdx.x;

  float sum = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) {
    sum += x[n * C + c];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) mean[c] = sdata[0] / static_cast<float>(N);
}

// =============================================================================
// Forward — compute per-feature variance
// =============================================================================

/**
 * @brief Compute the per-channel variance over the batch.
 *
 * Uses the mean from compute_mean_kernel.
 * var[c] = (1/N) · Σ_n (x[n*C + c] − mean[c])².
 *
 * @param x    Input activations (N × C)
 * @param mean Per-channel means [C]
 * @param var  Output per-channel variances [C]
 * @param N    Batch size
 * @param C    Number of features / channels
 */
__global__ void compute_variance_kernel(const float* __restrict__ x, const float* __restrict__ mean,
                                        float* __restrict__ var, int N, int C) {
  extern __shared__ float sdata[];
  int c = blockIdx.x;
  int tid = threadIdx.x;

  float mu = mean[c];
  float sum = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) {
    float diff = x[n * C + c] - mu;
    sum += diff * diff;
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) var[c] = sdata[0] / static_cast<float>(N);
}

// =============================================================================
// Forward — normalise, scale, shift
// =============================================================================

/**
 * @brief Batch-norm forward: normalise, scale, shift — y = γ · x̂ + β.
 *
 * Each thread handles one element (n, c).  Reads mean[c] and var[c]
 * computed in the preceding kernels, plus learnable γ[c], β[c].
 *
 * @param x     Input activations (N × C)
 * @param mean  Per-channel means [C]
 * @param var   Per-channel variances [C]
 * @param gamma Learnable scale parameter [C]
 * @param beta  Learnable shift parameter [C]
 * @param x_hat Output normalised activations (N × C)
 * @param y     Output transformed activations (N × C)
 * @param N     Batch size
 * @param C     Number of features / channels
 * @param eps   Small constant for numerical stability
 */
__global__ void batchnorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ mean,
                                         const float* __restrict__ var,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ x_hat,
                                         float* __restrict__ y, int N, int C, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C;
  if (idx >= total) return;

  int c = idx % C;
  float mu = mean[c];
  float inv = rsqrtf(var[c] + eps);
  float xh = (x[idx] - mu) * inv;
  x_hat[idx] = xh;
  y[idx] = gamma[c] * xh + beta[c];
}

// =============================================================================
// Forward — update running statistics (EMA)
// =============================================================================

/**
 * @brief EMA update of running statistics: running = (1−m)·running + m·batch.
 *
 * Called once per training step to maintain inference-time statistics.
 *
 * @param running  Running mean or variance to update [C]
 * @param batch    Current batch mean or variance [C]
 * @param C        Number of features / channels
 * @param momentum EMA momentum factor (typically 0.1)
 */
__global__ void update_running_stats_kernel(float* __restrict__ running,
                                            const float* __restrict__ batch, int C,
                                            float momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) running[c] = (1.0F - momentum) * running[c] + momentum * batch[c];
}

// =============================================================================
// Backward — compute dγ, dβ, dx
// =============================================================================

/**
 * @brief Compute gradients for gamma and beta via batch reduction.
 *
 * dγ[c] = Σ_n (dŷ[n,c] · x̂[n,c]) and dβ[c] = Σ_n dŷ[n,c].
 * One block per feature — same reduction pattern as the forward mean/var.
 *
 * @param dy     Upstream gradient (N × C)
 * @param x_hat  Normalised activations from forward pass (N × C)
 * @param dgamma Output gradient w.r.t. gamma [C]
 * @param dbeta  Output gradient w.r.t. beta [C]
 * @param N      Batch size
 * @param C      Number of features / channels
 */
__global__ void batchnorm_backward_dgamma_dbeta(const float* __restrict__ dy,
                                                const float* __restrict__ x_hat,
                                                float* __restrict__ dgamma,
                                                float* __restrict__ dbeta, int N, int C) {
  extern __shared__ float sdata[];  // two consecutive arrays of blockDim.x
  float* sg = sdata;
  float* sb = sdata + blockDim.x;

  int c = blockIdx.x;
  int tid = threadIdx.x;

  float sum_g = 0.0F;
  float sum_b = 0.0F;
  for (int n = tid; n < N; n += blockDim.x) {
    int idx = n * C + c;
    sum_g += dy[idx] * x_hat[idx];
    sum_b += dy[idx];
  }
  sg[tid] = sum_g;
  sb[tid] = sum_b;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sg[tid] += sg[tid + s];
      sb[tid] += sb[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    dgamma[c] = sg[0];
    dbeta[c] = sb[0];
  }
}

/**
 * @brief Compute gradient w.r.t. input x from upstream dŷ.
 *
 * dx = γ · inv_std · { dŷ − (1/N) · [dβ + x̂ · dγ] }.
 * Derived from differentiating through the normalisation graph.
 *
 * @param dy     Upstream gradient (N × C)
 * @param x_hat  Normalised activations from forward pass (N × C)
 * @param gamma  Learnable scale parameter [C]
 * @param var    Per-channel variances [C]
 * @param dgamma Pre-computed gradient w.r.t. gamma [C]
 * @param dbeta  Pre-computed gradient w.r.t. beta [C]
 * @param dx     Output gradient w.r.t. input (N × C)
 * @param N      Batch size
 * @param C      Number of features / channels
 * @param eps    Small constant for numerical stability
 */
__global__ void batchnorm_backward_dx(const float* __restrict__ dy, const float* __restrict__ x_hat,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ var,
                                      const float* __restrict__ dgamma,
                                      const float* __restrict__ dbeta, float* __restrict__ dx,
                                      int N, int C, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C;
  if (idx >= total) return;

  int c = idx % C;
  float inv = rsqrtf(var[c] + eps);
  float inv_N = 1.0F / static_cast<float>(N);

  // ---------- derivation of dx for batch normalisation ----------
  // Forward:  x_hat = (x - mu) / sigma     where sigma = sqrt(var + eps)
  // Both mu and var are functions of *all* x in the batch, so:
  //   dL/dx_i = gamma / sigma * [ dy_i
  //             - (1/N) * sum_j(dy_j)                    ... = dbeta / N
  //             - (1/N) * x_hat_i * sum_j(dy_j * x_hat_j) ... = x_hat * dgamma / N ]
  // Rearranging:  dx = gamma * inv * (dy - (1/N)(dbeta + x_hat * dgamma))
  // where inv = rsqrt(var + eps),  dbeta = sum(dy),  dgamma = sum(dy * x_hat).
  dx[idx] = gamma[c] * inv * (dy[idx] - inv_N * (dbeta[c] + x_hat[idx] * dgamma[c]));
}

// =============================================================================
// main — demonstrate batch normalization
// =============================================================================

int main() {
  constexpr int kN = 64;   // batch size
  constexpr int kC = 128;  // features
  constexpr float kEps = 1e-5F;
  constexpr float kMomentum = 0.1F;

  // -- Host data (random) --
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(3.0F, 5.0F);  // μ≈3, σ≈5 on purpose
  std::vector<float> h_x(kN * kC);
  for (auto& v : h_x) v = dist(gen);

  // Learnable params initialised to identity transform
  std::vector<float> h_gamma(kC, 1.0F);
  std::vector<float> h_beta(kC, 0.0F);

  // -- Allocate device buffers --
  float *d_x, *d_mean, *d_var, *d_gamma, *d_beta, *d_x_hat, *d_y;
  float *d_running_mean, *d_running_var;
  auto bytes_NC = static_cast<size_t>(kN * kC) * sizeof(float);
  auto bytes_C = static_cast<size_t>(kC) * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_x, bytes_NC));
  CUDA_CHECK(cudaMalloc(&d_x_hat, bytes_NC));
  CUDA_CHECK(cudaMalloc(&d_y, bytes_NC));
  CUDA_CHECK(cudaMalloc(&d_mean, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_var, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_gamma, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_beta, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_running_mean, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_running_var, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes_NC, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), bytes_C, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), bytes_C, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_running_mean, 0, bytes_C));
  // Running variance must be initialised to 1 (unit variance), NOT 0.
  // With EMA updates: running = (1-m)*running + m*batch, starting from 0
  // would bias early inference towards near-zero variance → exploding outputs.
  std::vector<float> ones_C(kC, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_running_var, ones_C.data(), bytes_C, cudaMemcpyHostToDevice));

  // -- Forward pass --
  int smem = kBlockSize * static_cast<int>(sizeof(float));
  compute_mean_kernel<<<kC, kBlockSize, smem>>>(d_x, d_mean, kN, kC);
  CUDA_CHECK(cudaGetLastError());
  compute_variance_kernel<<<kC, kBlockSize, smem>>>(d_x, d_mean, d_var, kN, kC);
  CUDA_CHECK(cudaGetLastError());

  int grid_elem = (kN * kC + kBlockSize - 1) / kBlockSize;
  batchnorm_forward_kernel<<<grid_elem, kBlockSize>>>(d_x, d_mean, d_var, d_gamma, d_beta, d_x_hat,
                                                      d_y, kN, kC, kEps);
  CUDA_CHECK(cudaGetLastError());

  int grid_C = (kC + kBlockSize - 1) / kBlockSize;
  update_running_stats_kernel<<<grid_C, kBlockSize>>>(d_running_mean, d_mean, kC, kMomentum);
  CUDA_CHECK(cudaGetLastError());
  update_running_stats_kernel<<<grid_C, kBlockSize>>>(d_running_var, d_var, kC, kMomentum);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());

  // -- Print statistics before and after BN --
  std::vector<float> h_mean(kC), h_var(kC), h_y(kN * kC);
  CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, bytes_C, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_var.data(), d_var, bytes_C, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes_NC, cudaMemcpyDeviceToHost));

  // Compute output stats for first 4 features
  std::printf("Batch Normalization — input was N(%g, %g²)\n", 3.0, 5.0);
  std::printf("%-8s %12s %12s %12s %12s\n", "Feature", "In μ", "In σ²", "Out μ", "Out σ²");
  for (int c = 0; c < 4; ++c) {
    double out_mean = 0.0;
    for (int n = 0; n < kN; ++n) out_mean += h_y[n * kC + c];
    out_mean /= kN;
    double out_var = 0.0;
    for (int n = 0; n < kN; ++n) {
      double diff = h_y[n * kC + c] - out_mean;
      out_var += diff * diff;
    }
    out_var /= kN;
    std::printf("%-8d %12.4f %12.4f %12.6f %12.6f\n", c, static_cast<double>(h_mean[c]),
                static_cast<double>(h_var[c]), out_mean, out_var);
  }

  // -- Backward pass (with synthetic upstream gradient = 1.0) --
  float *d_dy, *d_dgamma, *d_dbeta, *d_dx;
  CUDA_CHECK(cudaMalloc(&d_dy, bytes_NC));
  CUDA_CHECK(cudaMalloc(&d_dgamma, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_dbeta, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_dx, bytes_NC));

  std::vector<float> h_dy(kN * kC, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), bytes_NC, cudaMemcpyHostToDevice));

  int smem2 = 2 * kBlockSize * static_cast<int>(sizeof(float));
  batchnorm_backward_dgamma_dbeta<<<kC, kBlockSize, smem2>>>(d_dy, d_x_hat, d_dgamma, d_dbeta, kN,
                                                             kC);
  CUDA_CHECK(cudaGetLastError());
  batchnorm_backward_dx<<<grid_elem, kBlockSize>>>(d_dy, d_x_hat, d_gamma, d_var, d_dgamma, d_dbeta,
                                                   d_dx, kN, kC, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dgamma(kC), h_dbeta(kC);
  CUDA_CHECK(cudaMemcpy(h_dgamma.data(), d_dgamma, bytes_C, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dbeta.data(), d_dbeta, bytes_C, cudaMemcpyDeviceToHost));

  std::printf("\nBackward pass (upstream gradient = 1):\n");
  std::printf("  dγ[0..3] = %.4f, %.4f, %.4f, %.4f\n", static_cast<double>(h_dgamma[0]),
              static_cast<double>(h_dgamma[1]), static_cast<double>(h_dgamma[2]),
              static_cast<double>(h_dgamma[3]));
  std::printf("  dβ[0..3] = %.1f, %.1f, %.1f, %.1f\n", static_cast<double>(h_dbeta[0]),
              static_cast<double>(h_dbeta[1]), static_cast<double>(h_dbeta[2]),
              static_cast<double>(h_dbeta[3]));

  // -- Cleanup --
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_x_hat));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_var));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_running_mean));
  CUDA_CHECK(cudaFree(d_running_var));
  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dgamma));
  CUDA_CHECK(cudaFree(d_dbeta));
  CUDA_CHECK(cudaFree(d_dx));

  return EXIT_SUCCESS;
}
