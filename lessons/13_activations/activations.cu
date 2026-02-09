/**
 * @file activations.cu
 * @brief Lesson 13 — Activation Functions: ReLU, Sigmoid, Tanh, Softmax.
 *
 * Activation functions introduce **non-linearity** into a neural network.
 * Without them, a stack of dense layers (Y = X·W + b) collapses into a
 * single linear transformation — no matter how many layers you add.
 *
 * Each activation has a forward kernel and a backward kernel.  The backward
 * kernel computes the **local Jacobian** (element-wise derivative), which
 * is multiplied with the upstream gradient during backpropagation.
 *
 * ## Element-wise activations (ReLU, Sigmoid, Tanh)
 *
 * These are embarrassingly parallel: one thread per element, no shared
 * memory, no synchronisation.  They are **memory-bound** (1 load + 1 store
 * per thread, with 0–3 FLOPs).
 *
 * | Function | Forward f(x)           | Backward f'(x)              |
 * |----------|------------------------|-----------------------------|
 * | ReLU     | max(0, x)              | 1 if x > 0, else 0          |
 * | Sigmoid  | 1 / (1 + e^(-x))       | σ(x) · (1 − σ(x))            |
 * | Tanh     | tanh(x)                | 1 − tanh²(x)                |
 *
 * ## Row-wise activation (Softmax)
 *
 * Softmax maps each row (one sample, C classes) to a probability
 * distribution: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.
 *
 * Key implementation details:
 * - **Subtract the row max** before exp to prevent overflow (the result
 *   is mathematically identical but numerically stable).
 * - Uses **shared-memory reduction** to find the row max and sum.
 * - One block per row; threads within the block cooperate via shared mem.
 *
 * The backward pass for softmax is usually fused with the loss function
 * (cross-entropy), see Lesson 16.
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

// =============================================================================
// ReLU
// =============================================================================

/// @brief ReLU forward: out = max(0, in).
///
/// ReLU is the most popular activation in modern deep learning because:
/// 1. It's **fast** — just a comparison + conditional move.
/// 2. It doesn't saturate for positive inputs (unlike sigmoid/tanh),
///    so gradients flow freely through deep networks.
/// 3. It induces **sparsity** — roughly 50% of neurons output zero.
///
/// The downside is the "dying ReLU" problem: if a neuron's pre-activation
/// is always negative, its gradient is always zero and it never updates.
/// Variants like Leaky ReLU (f(x) = max(0.01x, x)) address this.
__global__ void relu_forward(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = fmaxf(0.0F, in[idx]);
}

/// @brief ReLU backward: grad_in = (in > 0) ? grad_out : 0.
///
/// The gradient of max(0, x) is a step function: 1 for x > 0, 0 otherwise.
/// This is the simplest example of the chain rule on the GPU: multiply the
/// upstream gradient (`grad_out`) by the local derivative.
__global__ void relu_backward(const float* in, const float* grad_out, float* grad_in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) grad_in[idx] = (in[idx] > 0.0F) ? grad_out[idx] : 0.0F;
}

// =============================================================================
// Sigmoid
// =============================================================================

/// @brief Sigmoid forward: out = 1 / (1 + exp(-in)).
///
/// Squashes any real value into (0, 1).  Historically used for binary
/// classification outputs.  In modern networks, sigmoid is mostly used in
/// gates (LSTM, attention) rather than hidden layers, because it **saturates**
/// for large |x| — the gradient approaches zero, slowing learning.
__global__ void sigmoid_forward(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 1.0F / (1.0F + expf(-in[idx]));
}

/// @brief Sigmoid backward: grad_in = grad_out * s * (1 - s).
///
/// Note that the backward takes `out` (not `in`) — the sigmoid value itself.
/// This avoids re-computing exp(-x).  The derivative σ'·(1−σ) has a maximum
/// of 0.25 at x = 0, so gradients are always attenuated through sigmoid.
__global__ void sigmoid_backward(const float* out, const float* grad_out, float* grad_in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float s = out[idx];
    grad_in[idx] = grad_out[idx] * s * (1.0F - s);
  }
}

// =============================================================================
// Tanh
// =============================================================================

/// @brief Tanh forward: out = tanh(in).
///
/// Maps inputs to (-1, 1).  Zero-centred (unlike sigmoid), which helps
/// gradient flow.  The CUDA intrinsic `tanhf()` uses a hardware-accelerated
/// approximation that is accurate to single-precision ULP.
__global__ void tanh_forward(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = tanhf(in[idx]);
}

/// @brief Tanh backward: grad_in = grad_out * (1 - t^2).
///
/// Like sigmoid, the backward uses the **output** `t = tanh(x)` to avoid
/// recomputing the forward.  The derivative 1 − t² has a peak of 1 at x = 0
/// and decays towards zero for large |x| (saturation).
__global__ void tanh_backward(const float* out, const float* grad_out, float* grad_in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float t = out[idx];
    grad_in[idx] = grad_out[idx] * (1.0F - t * t);
  }
}

// =============================================================================
// Softmax (row-wise, batch × classes)
// =============================================================================

/**
 * @brief Softmax forward: one block per row.
 *
 * ### Numerical stability trick
 * Naive softmax `exp(x_i) / sum(exp(x_j))` overflows for large x_i.
 * Subtracting the row max is mathematically equivalent:
 *   softmax(x)_i = exp(x_i - max) / sum(exp(x_j - max))
 * but keeps all exponents ≤ 0, preventing overflow.
 *
 * ### Parallelism
 * This kernel assigns **one block per row** (= one sample in the batch).
 * Threads within the block cooperate to find the max, compute exponentials,
 * sum them, and normalise.  Shared-memory reductions (same pattern as
 * Lesson 08) are used for the max and sum operations.
 *
 * For very wide rows (many classes), multiple threads per class handle
 * the strided loop, and the reduction combines their partial results.
 */
__global__ void softmax_forward(const float* in, float* out, int cols) {
  extern __shared__ float sdata[];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;

  // Find row max
  float max_val = -1e30F;
  for (int c = tid; c < cols; c += blockDim.x) {
    max_val = fmaxf(max_val, row_in[c]);
  }
  sdata[tid] = max_val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  float row_max = sdata[0];

  // Compute exp and sum
  float local_sum = 0.0F;
  for (int c = tid; c < cols; c += blockDim.x) {
    float e = expf(row_in[c] - row_max);
    row_out[c] = e;
    local_sum += e;
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float total = sdata[0];

  // Normalize
  for (int c = tid; c < cols; c += blockDim.x) {
    row_out[c] /= total;
  }
}

int main() {
  constexpr int kN = 8;
  std::vector<float> h_in = {-2, -1, 0, 0.5, 1, 2, 3, 4};
  std::vector<float> h_out(kN);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, kN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

  // ReLU
  relu_forward<<<1, kN>>>(d_in, d_out, kN);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kN * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("ReLU:    ");
  for (auto v : h_out) std::printf("%.2f ", static_cast<double>(v));
  std::printf("\n");

  // Sigmoid
  sigmoid_forward<<<1, kN>>>(d_in, d_out, kN);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kN * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Sigmoid: ");
  for (auto v : h_out) std::printf("%.4f ", static_cast<double>(v));
  std::printf("\n");

  // Tanh
  tanh_forward<<<1, kN>>>(d_in, d_out, kN);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kN * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Tanh:    ");
  for (auto v : h_out) std::printf("%.4f ", static_cast<double>(v));
  std::printf("\n");

  // Softmax (treat as 1 row of 8)
  int smem = 256 * sizeof(float);
  softmax_forward<<<1, 256, smem>>>(d_in, d_out, kN);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kN * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Softmax: ");
  for (auto v : h_out) std::printf("%.4f ", static_cast<double>(v));
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
