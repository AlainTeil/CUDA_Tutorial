/**
 * @file loss_test.cu
 * @brief Unit tests for Lesson 16 — MSE and Cross-Entropy losses.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) FAIL() << "CUDA error: " << cudaGetErrorString(err_); \
  } while (0)

// Kernel definitions (self-contained) -----------------------------------------

__global__ void mse_forward(const float* pred, const float* target, float* diff_sq, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float d = pred[i] - target[i];
    diff_sq[i] = d * d;
  }
}

__global__ void mse_backward(const float* pred, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = 2.0F * (pred[i] - target[i]) / static_cast<float>(N);
}

__global__ void log_softmax(const float* logits, float* log_sm, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float val = (tid < N) ? logits[tid] : -1e30F;
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();
  float exp_val = (tid < N) ? expf(val - max_val) : 0.0F;
  sdata[tid] = exp_val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float log_sum = logf(sdata[0]);
  if (tid < N) log_sm[tid] = (val - max_val) - log_sum;
}

__global__ void cross_entropy_forward(const float* log_sm, const float* target, float* elem_loss,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem_loss[i] = -target[i] * log_sm[i];
}

__global__ void cross_entropy_backward(const float* log_sm, const float* target, float* grad,
                                       int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

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

// Helpers ---------------------------------------------------------------------

static int next_pow2(int n) {
  int v = 1;
  while (v < n) v <<= 1;
  return v;
}

// =============================================================================
// MSE — forward
// =============================================================================

TEST(MSETest, ForwardValue) {
  constexpr int N = 4;
  std::vector<float> h_pred = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> h_target = {1.5F, 2.5F, 3.5F, 4.5F};

  float *d_pred, *d_target, *d_diff, *d_loss;
  CUDA_CHECK(cudaMalloc(&d_pred, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_diff, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_pred, h_pred.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  mse_forward<<<1, N>>>(d_pred, d_target, d_diff, N);
  int bp = next_pow2(N);
  reduce_sum<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(d_diff, d_loss, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_sum;
  CUDA_CHECK(cudaMemcpy(&h_sum, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  float mse = h_sum / static_cast<float>(N);

  // Expected: mean of 0.25 each → 0.25
  EXPECT_NEAR(mse, 0.25F, 1e-5F);

  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_diff));
  CUDA_CHECK(cudaFree(d_loss));
}

// =============================================================================
// MSE — backward gradient check
// =============================================================================

TEST(MSETest, BackwardGradientCheck) {
  constexpr int N = 5;
  std::vector<float> h_pred = {0.2F, 0.8F, -0.5F, 1.3F, 0.0F};
  std::vector<float> h_target = {0.0F, 1.0F, -1.0F, 1.0F, 0.5F};

  float *d_pred, *d_target, *d_grad;
  CUDA_CHECK(cudaMalloc(&d_pred, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_pred, h_pred.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  mse_backward<<<1, N>>>(d_pred, d_target, d_grad, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad(N);
  CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Finite-difference check
  constexpr float eps = 1e-3F;
  for (int i = 0; i < N; ++i) {
    auto compute_mse = [&](std::vector<float>& p) {
      float s = 0.0F;
      for (int j = 0; j < N; ++j)
        s += (p[static_cast<size_t>(j)] - h_target[static_cast<size_t>(j)]) *
             (p[static_cast<size_t>(j)] - h_target[static_cast<size_t>(j)]);
      return s / static_cast<float>(N);
    };
    auto pp = h_pred;
    auto pm = h_pred;
    pp[static_cast<size_t>(i)] += eps;
    pm[static_cast<size_t>(i)] -= eps;
    float fd = (compute_mse(pp) - compute_mse(pm)) / (2.0F * eps);
    EXPECT_NEAR(h_grad[static_cast<size_t>(i)], fd, 1e-3F) << "i=" << i;
  }

  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_grad));
}

// =============================================================================
// Log-Softmax — sums to ~0 and exp sums to ~1
// =============================================================================

TEST(LogSoftmaxTest, Properties) {
  constexpr int N = 5;
  std::vector<float> h_input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  int bp = next_pow2(N);
  log_softmax<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(d_in, d_out, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(N);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  // All log-softmax values should be <= 0
  for (int i = 0; i < N; ++i) EXPECT_LE(h_out[static_cast<size_t>(i)], 0.0F);

  // exp(log_softmax) should sum to 1
  float esum = 0.0F;
  for (int i = 0; i < N; ++i) esum += std::exp(h_out[static_cast<size_t>(i)]);
  EXPECT_NEAR(esum, 1.0F, 1e-5F);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

// =============================================================================
// Cross-Entropy — known value
// =============================================================================

TEST(CrossEntropyTest, KnownValue) {
  constexpr int N = 3;
  // logits [0, 0, 0] → softmax = [1/3,1/3,1/3] → CE with target [0,0,1] = -log(1/3) ≈ 1.0986
  std::vector<float> h_logits = {0.0F, 0.0F, 0.0F};
  std::vector<float> h_target = {0.0F, 0.0F, 1.0F};

  float *d_logits, *d_target, *d_log_sm, *d_elem, *d_loss;
  CUDA_CHECK(cudaMalloc(&d_logits, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_log_sm, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_elem, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  int bp = next_pow2(N);
  log_softmax<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(d_logits, d_log_sm, N);
  cross_entropy_forward<<<1, N>>>(d_log_sm, d_target, d_elem, N);
  reduce_sum<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(d_elem, d_loss, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_loss;
  CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(h_loss, std::log(3.0F), 1e-4F);

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_log_sm));
  CUDA_CHECK(cudaFree(d_elem));
  CUDA_CHECK(cudaFree(d_loss));
}

// =============================================================================
// Cross-Entropy — backward gradient check
// =============================================================================

TEST(CrossEntropyTest, BackwardGradientCheck) {
  constexpr int N = 4;
  std::vector<float> h_logits = {1.0F, 2.0F, 5.0F, 0.5F};
  std::vector<float> h_target = {0.0F, 0.0F, 1.0F, 0.0F};

  float *d_logits, *d_target, *d_log_sm, *d_grad;
  CUDA_CHECK(cudaMalloc(&d_logits, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_log_sm, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  int bp = next_pow2(N);
  log_softmax<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(d_logits, d_log_sm, N);
  cross_entropy_backward<<<1, N>>>(d_log_sm, d_target, d_grad, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad(N);
  CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Gradient should sum to 0 (softmax - target sums to 1-1=0)
  float gsum = std::accumulate(h_grad.begin(), h_grad.end(), 0.0F);
  EXPECT_NEAR(gsum, 0.0F, 1e-5F);

  // Finite-difference
  constexpr float eps = 1e-3F;
  auto compute_ce = [&](std::vector<float>& logits) {
    float mx = *std::max_element(logits.begin(), logits.end());
    float esum = 0.0F;
    for (int j = 0; j < N; ++j) esum += std::exp(logits[static_cast<size_t>(j)] - mx);
    float lse = mx + std::log(esum);
    float loss = 0.0F;
    for (int j = 0; j < N; ++j)
      loss -= h_target[static_cast<size_t>(j)] * (logits[static_cast<size_t>(j)] - lse);
    return loss;
  };
  for (int i = 0; i < N; ++i) {
    auto lp = h_logits;
    auto lm = h_logits;
    lp[static_cast<size_t>(i)] += eps;
    lm[static_cast<size_t>(i)] -= eps;
    float fd = (compute_ce(lp) - compute_ce(lm)) / (2.0F * eps);
    EXPECT_NEAR(h_grad[static_cast<size_t>(i)], fd, 1e-3F) << "i=" << i;
  }

  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_log_sm));
  CUDA_CHECK(cudaFree(d_grad));
}
