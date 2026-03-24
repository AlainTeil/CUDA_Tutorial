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
    const cudaError_t err_ = (call);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// =============================================================================
// MSE loss:  L = (1/N) * sum_i (pred_i - target_i)^2
// =============================================================================

/**
 * @brief MSE forward: compute per-element squared difference (pred − target)².
 *
 * Writes per-element squared differences.  A separate reduce-sum kernel
 * totals them and divides by N to complete the mean.
 *
 * @param pred    Predicted values [N]
 * @param target  Ground-truth values [N]
 * @param diff_sq Output per-element squared differences [N]
 * @param N       Number of elements
 */
__global__ void mse_forward(const float* pred, const float* target, float* diff_sq, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float d = pred[i] - target[i];
    diff_sq[i] = d * d;
  }
}

/**
 * @brief MSE backward: compute gradient dL/dpred = 2(pred − target) / N.
 *
 * Element-wise gradient — each prediction's gradient depends only on its
 * own error, making MSE embarrassingly parallel.
 *
 * @param pred   Predicted values [N]
 * @param target Ground-truth values [N]
 * @param grad   Output gradient w.r.t. predictions [N]
 * @param N      Number of elements
 */
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

/**
 * @brief Numerically stable log-softmax using the subtract-max trick.
 *
 * Computes log_softmax_i = (x_i − max) − log(Σ exp(x_j − max)) using
 * shared-memory parallel reductions.  blockDim.x must be a power-of-two
 * ≥ N; out-of-range threads contribute sentinels so reductions stay correct.
 *
 * @param logits Input raw logits [N]
 * @param log_sm Output log-softmax values [N]
 * @param N      Number of classes
 */
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

/**
 * @brief Cross-entropy forward: compute per-element loss −target · log_softmax.
 *
 * For one-hot targets only the true-class term survives, but the dot-product
 * form keeps the kernel general (e.g. label smoothing).  A reduce-sum
 * kernel totals the per-element losses.
 *
 * @param log_sm    Log-softmax values [N]
 * @param target    Target distribution [N]
 * @param elem_loss Output per-element losses [N]
 * @param N         Number of classes
 */
__global__ void cross_entropy_forward(const float* log_sm, const float* target, float* elem_loss,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    elem_loss[i] = -target[i] * log_sm[i];
  }
}

/**
 * @brief Cross-entropy backward: dL/d(logit) = softmax − target.
 *
 * Recovers softmax from stored log-softmax via exp(log_sm).  The simple
 * gradient form is why softmax + CE are always used together.
 *
 * @param log_sm Log-softmax values [N]
 * @param target Target distribution [N]
 * @param grad   Output gradient w.r.t. logits [N]
 * @param N      Number of classes
 */
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

/**
 * @brief Parallel sum reduction for loss aggregation.
 *
 * Sums N floats using a shared-memory tree reduction and writes the
 * result to out[0].  Single-block kernel; blockDim.x must be ≥ N.
 *
 * @param in  Input array to sum [N]
 * @param out Output scalar (single element)
 * @param N   Number of elements to sum
 */
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
  CUDA_CHECK(cudaGetLastError());
  cross_entropy_forward<<<1, N>>>(d_log_sm, d_target, d_elem, N);
  CUDA_CHECK(cudaGetLastError());
  reduce_sum<<<1, blockSize, static_cast<size_t>(blockSize) * sizeof(float)>>>(d_elem, d_loss, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_loss;
  CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Cross-entropy loss: %.6f\n", static_cast<double>(h_loss));

  cross_entropy_backward<<<1, N>>>(d_log_sm, d_target, d_grad, N);
  CUDA_CHECK(cudaGetLastError());
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
