/**
 * @file loss.cu
 * @brief Lesson 16 — Loss Functions for Deep Learning.
 *
 * A loss function maps a model’s predictions to a **single scalar** that
 * measures how far those predictions are from the true labels.  Because
 * gradient descent needs a scalar to differentiate, the loss is the
 * starting point of every backward pass.
 *
 * This lesson implements two workhorse losses:
 *
 *  1. **Mean Squared Error (MSE)** — natural for regression tasks.
 *     L = (1/N) Σ (pred_i − target_i)²
 *
 *  2. **Cross-Entropy (CE) with log-softmax** — the standard for
 *     multi-class classification.  We fuse log and softmax into a single
 *     numerically stable operation (see notes on log_softmax below).
 *     L = −Σ target_i · log(softmax(logit_i))
 *
 * All kernels operate on a single sample of size N (number of classes).
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
// MSE loss:  L = (1/N) * sum_i (pred_i - target_i)^2
// =============================================================================

/// Forward: writes per-element squared diff.  A separate reduce-sum kernel
/// (or host-side sum) totals them and divides by N.
///
/// MSE is the go-to loss for regression (predicting continuous values).
/// It penalises large errors quadratically, making it sensitive to outliers.
__global__ void mse_forward(const float* pred, const float* target, float* diff_sq, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float d = pred[i] - target[i];
    diff_sq[i] = d * d;
  }
}

/// Backward:  dL/dpred_i = (2/N) * (pred_i - target_i)
///
/// This is a simple element-wise gradient — each prediction’s gradient
/// depends only on its own error, making MSE embarrassingly parallel.
__global__ void mse_backward(const float* pred, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    grad[i] = 2.0F * (pred[i] - target[i]) / static_cast<float>(N);
  }
}

// =============================================================================
// Log-Softmax:                 log_softmax_i = x_i - log(sum_j exp(x_j))
// Cross-Entropy loss (labels): L = -sum_i target_i * log_softmax_i
// =============================================================================

/// Numerically stable log-softmax using the "subtract-max" trick.
///
/// Naïvely computing  log(exp(x_i) / Σ exp(x_j))  overflows when any x_j
/// is large (e.g. 90 → exp(90) ≈ 1.2e39, above FP32 max ≈ 3.4e38).
///
/// The identity  softmax(x) = softmax(x − c)  for any constant c lets us
/// pick c = max(x).  After subtraction the largest exponent is exp(0) = 1,
/// so no term overflows.  We then compute:
///     log_softmax_i = (x_i − max) − log(Σ exp(x_j − max))
///
/// Implementation note: this kernel uses **shared-memory parallel
/// reductions** to find the max and the sum.  blockDim.x must be a
/// power-of-two ≥ N.  Threads with tid ≥ N contribute a sentinel (−1e30
/// for max, 0 for sum) so the reduction stays correct.
__global__ void log_softmax(const float* logits, float* log_sm, int N) {
  // Single-block kernel — N is the number of classes (typically small).
  extern __shared__ float sdata[];

  int tid = threadIdx.x;

  // Load into shared memory; out-of-range threads get a very negative value
  // so they never win the max reduction.
  float val = (tid < N) ? logits[tid] : -1e30F;
  sdata[tid] = val;
  __syncthreads();

  // --- Step 1: parallel reduction to find max ---
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float max_val = sdata[0];  // broadcast to all threads
  __syncthreads();

  // --- Step 2: exp(x - max) and parallel sum ---
  float exp_val = (tid < N) ? expf(val - max_val) : 0.0F;
  sdata[tid] = exp_val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float log_sum = logf(sdata[0]);  // log(Σ exp(x_j − max))
  __syncthreads();

  if (tid < N) {
    log_sm[tid] = (val - max_val) - log_sum;
  }
}

/// Cross-entropy forward: L = -Σ target_i · log_softmax_i
///
/// For one-hot targets only the true-class term survives, but writing it
/// as a dot product keeps the kernel general (e.g. label smoothing).
///
/// Produces per-element losses; a reduce-sum kernel totals them.
__global__ void cross_entropy_forward(const float* log_sm, const float* target, float* elem_loss,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    elem_loss[i] = -target[i] * log_sm[i];
  }
}

/// Cross-entropy backward (combined with softmax):
///    dL/d(logit_i) = softmax_i − target_i
///
/// This remarkably simple gradient is the reason softmax + CE are always
/// used together.  The derivation uses the quotient rule on softmax and
/// the chain rule through −log; most terms cancel, leaving softmax − target.
///
/// Since we stored log_softmax, we recover softmax as exp(log_sm[i]).
__global__ void cross_entropy_backward(const float* log_sm, const float* target, float* grad,
                                       int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    grad[i] = expf(log_sm[i]) - target[i];
  }
}

// =============================================================================
// Simple reduce-sum helper (single block)
// =============================================================================

/// Sums N floats using a shared-memory tree reduction.  Returns the result
/// in out[0].  This is kept as a separate kernel so the per-element loss
/// computation and the summation can be launched independently — a common
/// pattern in GPU programming where we prefer many simple kernels over one
/// complex monolithic kernel.
__global__ void reduce_sum(const float* in, float* out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  sdata[tid] = (tid < N) ? in[tid] : 0.0F;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) *out = sdata[0];
}

// =============================================================================
// Demo
// =============================================================================

int main() {
  constexpr int N = 4;

  // Predictions (logits) and one-hot target (class 2).
  // Logits are the raw (un-normalised) outputs of the network — log-softmax
  // will convert them to a valid log-probability distribution.
  std::vector<float> h_logits = {1.0F, 2.0F, 5.0F, 0.5F};
  std::vector<float> h_target = {0.0F, 0.0F, 1.0F, 0.0F};

  float *d_logits, *d_target, *d_log_sm, *d_elem, *d_loss, *d_grad;
  CUDA_CHECK(cudaMalloc(&d_logits, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_log_sm, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_elem, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Block size must be a power-of-two for the tree reductions inside
  // log_softmax and reduce_sum to work correctly.
  int blockSize = 1;
  while (blockSize < N) blockSize <<= 1;

  log_softmax<<<1, blockSize, static_cast<size_t>(blockSize) * sizeof(float)>>>(d_logits, d_log_sm,
                                                                                N);
  cross_entropy_forward<<<1, N>>>(d_log_sm, d_target, d_elem, N);
  reduce_sum<<<1, blockSize, static_cast<size_t>(blockSize) * sizeof(float)>>>(d_elem, d_loss, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_loss;
  CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Cross-entropy loss: %.6f\n", static_cast<double>(h_loss));

  cross_entropy_backward<<<1, N>>>(d_log_sm, d_target, d_grad, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad(N);
  CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Gradients:");
  for (int i = 0; i < N; ++i)
    std::printf(" %.6f", static_cast<double>(h_grad[static_cast<size_t>(i)]));
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_log_sm));
  CUDA_CHECK(cudaFree(d_elem));
  CUDA_CHECK(cudaFree(d_loss));
  CUDA_CHECK(cudaFree(d_grad));
  return EXIT_SUCCESS;
}
