/**
 * @file dropout.cu
 * @brief Lesson 24 — Dropout Regularization.
 *
 * **Dropout** (Srivastava et al., 2014) is a powerful regularisation
 * technique that randomly zeroes activations with probability **p** during
 * training.  This prevents neurons from co-adapting — each neuron must
 * learn features that are useful in conjunction with many random subsets of
 * other neurons, which results in more robust representations.
 *
 * ## Inverted Dropout
 *
 * The standard "inverted" variant scales surviving activations by 1/(1−p)
 * during **training** so that the expected value is preserved:
 *
 *     y_i = { x_i / (1 − p)    with probability 1 − p   (keep)
 *           { 0                 with probability p        (drop)
 *
 * At **inference** time dropout is disabled (identity pass-through) and **no
 * scaling** is needed because the training-time scaling already compensated.
 *
 * ## CUDA implementation notes
 *
 * We need per-element random decisions (Bernoulli draws).  Instead of using
 * the full cuRAND device API we employ a lightweight hash-based PRNG:
 *
 *     hash(seed, idx) → [0, 1)    (Philox-style hash for illustration)
 *
 * This keeps the lesson dependency-free (no cuRAND library link).  In
 * production code you would use `curandStatePhilox4_32_10_t` from
 * `<curand_kernel.h>` which ships with every CUDA Toolkit install as a
 * header-only library.
 *
 * The mask generated in the forward pass is **stored** and re-used in the
 * backward pass — this is critical: the same elements that were zeroed in
 * forward must have zero gradient in backward.
 *
 * See Lesson 13 for other element-wise operations (ReLU, etc.) and
 * Lesson 23 for Batch Normalisation, another regularisation technique.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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
// Hash-based PRNG helpers
// =============================================================================

/// @brief A simple hash that maps (seed, index) to a pseudo-random uint32.
///
/// This is a reduced-round Philox-like hash.  It is **not** cryptographic
/// but provides sufficient statistical quality for dropout masks.  In
/// production, prefer cuRAND's Philox4x32 counter-based generator.
__device__ __forceinline__ unsigned int hash_index(unsigned long long seed, int idx) {
  unsigned long long h = seed ^ (static_cast<unsigned long long>(idx) * 2654435761ULL);
  h ^= h >> 17;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 31;
  h *= 0x94d049bb133111ebULL;
  h ^= h >> 32;
  return static_cast<unsigned int>(h);
}

/// @brief Convert hash to float in [0, 1).
__device__ __forceinline__ float hash_to_uniform(unsigned int h) {
  return static_cast<float>(h & 0x7FFFFFU) / static_cast<float>(0x800000U);
}

// =============================================================================
// Dropout forward (training mode)
// =============================================================================

/**
 * @brief Dropout forward: generate mask via PRNG, apply inverted scaling.
 *
 * Zeros elements with probability p and scales surviving activations
 * by 1/(1−p).  Stores the binary mask for the backward pass.
 *
 * @param x    Input activations [n]
 * @param y    Output activations [n]
 * @param mask Output binary mask [n] (1 = keep, 0 = drop)
 * @param n    Number of elements
 * @param p    Drop probability (0 < p < 1)
 * @param seed Random seed for the hash-based PRNG
 */
__global__ void dropout_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                       float* __restrict__ mask, int n, float p,
                                       unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  unsigned int h = hash_index(seed, idx);
  float u = hash_to_uniform(h);
  float keep = (u >= p) ? 1.0F : 0.0F;
  float scale = 1.0F / (1.0F - p);

  mask[idx] = keep;
  y[idx] = x[idx] * keep * scale;
}

// =============================================================================
// Dropout backward
// =============================================================================

/**
 * @brief Dropout backward: re-apply the forward mask to the gradient.
 *
 * grad_in[i] = grad_out[i] * mask[i] / (1 − p).  The same mask from the
 * forward pass is reused so zeroed elements receive zero gradient.
 *
 * @param grad_out Upstream gradient [n]
 * @param mask     Binary mask from the forward pass [n]
 * @param grad_in  Output gradient w.r.t. input [n]
 * @param n        Number of elements
 * @param p        Drop probability (same as forward)
 */
__global__ void dropout_backward_kernel(const float* __restrict__ grad_out,
                                        const float* __restrict__ mask, float* __restrict__ grad_in,
                                        int n, float p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float scale = 1.0F / (1.0F - p);
  grad_in[idx] = grad_out[idx] * mask[idx] * scale;
}

// =============================================================================
// Dropout inference (identity)
// =============================================================================

/**
 * @brief Inference-mode dropout: identity pass-through (no dropout).
 *
 * Because inverted dropout scales at training time, no adjustment is
 * needed at inference — the input is simply copied to the output.
 *
 * @param x Input activations [n]
 * @param y Output activations [n]
 * @param n Number of elements
 */
__global__ void dropout_inference_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx];
}

// =============================================================================
// main
// =============================================================================

int main() {
  constexpr int kN = 1024;
  constexpr float kP = 0.5F;  // drop probability
  constexpr unsigned long long kSeed = 12345ULL;

  // Generate random input
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> h_x(kN);
  for (auto& v : h_x) v = dist(gen);

  float *d_x, *d_y, *d_mask;
  auto bytes = static_cast<size_t>(kN) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_mask, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  int grid = (kN + kBlockSize - 1) / kBlockSize;
  dropout_forward_kernel<<<grid, kBlockSize>>>(d_x, d_y, d_mask, kN, kP, kSeed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(kN), h_mask(kN);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, bytes, cudaMemcpyDeviceToHost));

  // Count zeros and non-zeros
  int zeros = 0;
  for (auto v : h_mask) {
    if (v == 0.0F) ++zeros;
  }
  std::printf("Dropout (p=%.1f): %d / %d elements dropped (%.1f%%)\n", static_cast<double>(kP),
              zeros, kN, 100.0 * zeros / static_cast<double>(kN));

  // Show first 10 values
  std::printf("\n%6s %10s %6s %10s\n", "idx", "x", "mask", "y");
  for (int i = 0; i < 10; ++i) {
    std::printf("%6d %10.4f %6.0f %10.4f\n", i, static_cast<double>(h_x[static_cast<size_t>(i)]),
                static_cast<double>(h_mask[static_cast<size_t>(i)]),
                static_cast<double>(h_y[static_cast<size_t>(i)]));
  }

  // Backward pass
  float *d_go, *d_gi;
  CUDA_CHECK(cudaMalloc(&d_go, bytes));
  CUDA_CHECK(cudaMalloc(&d_gi, bytes));
  // Upstream gradient = 1
  std::vector<float> h_go(kN, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_go, h_go.data(), bytes, cudaMemcpyHostToDevice));

  dropout_backward_kernel<<<grid, kBlockSize>>>(d_go, d_mask, d_gi, kN, kP);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_gi(kN);
  CUDA_CHECK(cudaMemcpy(h_gi.data(), d_gi, bytes, cudaMemcpyDeviceToHost));

  std::printf("\nBackward (grad_out=1): first 10 grad_in values:\n  ");
  for (int i = 0; i < 10; ++i)
    std::printf("%.2f ", static_cast<double>(h_gi[static_cast<size_t>(i)]));
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_mask));
  CUDA_CHECK(cudaFree(d_go));
  CUDA_CHECK(cudaFree(d_gi));

  return EXIT_SUCCESS;
}
