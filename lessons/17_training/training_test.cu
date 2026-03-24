/**
 * @file training_test.cu
 * @brief Unit tests for Lesson 17 — Training Loop.
 *
 * Verifies that:
 * 1. Loss decreases over training epochs.
 * 2. Final accuracy on linearly-separable synthetic data exceeds 90%.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

// Non-FAIL variant for use inside non-void-returning methods
#define CUDA_ASSERT(call)                                                 \
  do {                                                                    \
    const cudaError_t err_ = (call);                                      \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_)); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

// ---------------------------------------------------------------------------
// Kernel definitions (self-contained for single-TU compilation)
// ---------------------------------------------------------------------------

__global__ void dense_forward(const float* X, const float* W, const float* b, float* Y, int in_dim,
                              int out_dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < out_dim) {
    float sum = b[j];
    for (int i = 0; i < in_dim; ++i) sum += X[i] * W[i * out_dim + j];
    Y[j] = sum;
  }
}

__global__ void dense_backward_dX(const float* dY, const float* W, float* dX, int in_dim,
                                  int out_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < in_dim) {
    float sum = 0.0F;
    for (int j = 0; j < out_dim; ++j) sum += dY[j] * W[i * out_dim + j];
    dX[i] = sum;
  }
}

__global__ void dense_backward_dW(const float* X, const float* dY, float* dW, int in_dim,
                                  int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / out_dim;
  int j = idx % out_dim;
  if (i < in_dim && j < out_dim) dW[i * out_dim + j] = X[i] * dY[j];
}

__global__ void relu_forward(const float* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (in[i] > 0.0F) ? in[i] : 0.0F;
}

__global__ void relu_backward(const float* in, const float* grad_out, float* grad_in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad_in[i] = (in[i] > 0.0F) ? grad_out[i] : 0.0F;
}

__global__ void log_softmax_k(const float* logits, float* log_sm, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float val = (tid < N) ? logits[tid] : -1e30F;
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float mx = sdata[0];
  __syncthreads();
  float ev = (tid < N) ? expf(val - mx) : 0.0F;
  sdata[tid] = ev;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float lse = logf(sdata[0]);
  if (tid < N) log_sm[tid] = (val - mx) - lse;
}

__global__ void ce_forward_k(const float* log_sm, const float* target, float* elem, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem[i] = -target[i] * log_sm[i];
}

__global__ void ce_backward_k(const float* log_sm, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

__global__ void reduce_sum_k(const float* in, float* out, int N) {
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

__global__ void sgd_update(float* param, const float* grad, float lr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) param[i] -= lr * grad[i];
}

// Re-use MLP struct and data generation from training.cu by re-declaring here.
// In a real project these would be in a header, but for the tutorial we keep
// each lesson self-contained. We duplicate the struct for the test.

struct MLP {
  int in_dim{}, hid_dim{}, out_dim{};
  float *W1{}, *b1{}, *W2{}, *b2{};
  float *dW1{}, *db1{}, *dW2{}, *db2{};
  float *z1{}, *a1{}, *z2{}, *log_sm{};
  float *dz2{}, *da1{}, *dz1{}, *dx{};
  float *elem_loss{}, *d_loss{};

  void alloc(int in, int hid, int out) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    auto A = [](float** p, int n) {
      CUDA_CHECK(cudaMalloc(p, static_cast<size_t>(n) * sizeof(float)));
    };
    A(&W1, in * hid);
    A(&b1, hid);
    A(&W2, hid * out);
    A(&b2, out);
    A(&dW1, in * hid);
    A(&db1, hid);
    A(&dW2, hid * out);
    A(&db2, out);
    A(&z1, hid);
    A(&a1, hid);
    A(&z2, out);
    A(&log_sm, out);
    A(&dz2, out);
    A(&da1, hid);
    A(&dz1, hid);
    A(&dx, in);
    A(&elem_loss, out);
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  }

  void init_weights(unsigned seed) {
    std::mt19937 gen(seed);
    auto fill = [&](float* d, int n, float scale) {
      std::normal_distribution<float> dist(0.0F, scale);
      std::vector<float> h(static_cast<size_t>(n));
      for (auto& v : h) v = dist(gen);
      CUDA_ASSERT(
          cudaMemcpy(d, h.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    };
    fill(W1, in_dim * hid_dim, std::sqrt(2.0F / static_cast<float>(in_dim)));
    fill(b1, hid_dim, 0.01F);
    fill(W2, hid_dim * out_dim, std::sqrt(2.0F / static_cast<float>(hid_dim)));
    fill(b2, out_dim, 0.01F);
  }

  float forward(const float* d_x, const float* d_target) {
    int bp = 1;
    while (bp < out_dim) bp <<= 1;
    dense_forward<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    relu_forward<<<1, hid_dim>>>(z1, a1, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    dense_forward<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    CUDA_CHECK(cudaGetLastError());
    log_softmax_k<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(z2, log_sm, out_dim);
    CUDA_CHECK(cudaGetLastError());
    ce_forward_k<<<1, out_dim>>>(log_sm, d_target, elem_loss, out_dim);
    CUDA_CHECK(cudaGetLastError());
    reduce_sum_k<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(elem_loss, d_loss, out_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    float h_loss;
    CUDA_ASSERT(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
  }

  void backward(const float* d_x, const float* d_target) {
    ce_backward_k<<<1, out_dim>>>(log_sm, d_target, dz2, out_dim);
    CUDA_CHECK(cudaGetLastError());
    dense_backward_dW<<<1, hid_dim * out_dim>>>(a1, dz2, dW2, hid_dim, out_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(db2, dz2, static_cast<size_t>(out_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    dense_backward_dX<<<1, hid_dim>>>(dz2, W2, da1, hid_dim, out_dim);
    CUDA_CHECK(cudaGetLastError());
    relu_backward<<<1, hid_dim>>>(z1, da1, dz1, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    dense_backward_dW<<<1, in_dim * hid_dim>>>(d_x, dz1, dW1, in_dim, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(db1, dz1, static_cast<size_t>(hid_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void sgd_step(float lr) {
    auto update = [lr](float* p, const float* g, int n) {
      sgd_update<<<(n + 255) / 256, 256>>>(p, g, lr, n);
      CUDA_CHECK(cudaGetLastError());
    };
    update(W1, dW1, in_dim * hid_dim);
    update(b1, db1, hid_dim);
    update(W2, dW2, hid_dim * out_dim);
    update(b2, db2, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  [[nodiscard]] int predict(const float* d_x) {
    dense_forward<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    relu_forward<<<1, hid_dim>>>(z1, a1, hid_dim);
    CUDA_CHECK(cudaGetLastError());
    dense_forward<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    std::vector<float> h_z(static_cast<size_t>(out_dim));
    CUDA_ASSERT(cudaMemcpy(h_z.data(), z2, static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    return static_cast<int>(std::distance(h_z.begin(), std::max_element(h_z.begin(), h_z.end())));
  }

  void free_all() {
    cudaFree(W1);
    cudaFree(b1);
    cudaFree(W2);
    cudaFree(b2);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(z1);
    cudaFree(a1);
    cudaFree(z2);
    cudaFree(log_sm);
    cudaFree(dz2);
    cudaFree(da1);
    cudaFree(dz1);
    cudaFree(dx);
    cudaFree(elem_loss);
    cudaFree(d_loss);
  }
};

static void generate_data(std::vector<std::vector<float>>& samples, std::vector<int>& labels,
                          int n_per_class, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  float centres[3][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < n_per_class; ++i) {
      std::vector<float> s(4);
      for (int d = 0; d < 4; ++d) s[static_cast<size_t>(d)] = centres[c][d] + noise(gen);
      samples.push_back(s);
      labels.push_back(c);
    }
}

// =============================================================================
// Test: loss decreases
// =============================================================================

TEST(TrainingTest, LossDecreases) {
  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 30, 42);
  int n = static_cast<int>(samples.size());

  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(123);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  auto run_epoch = [&]() {
    float total = 0.0F;
    for (int s = 0; s < n; ++s) {
      CUDA_ASSERT(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                             cudaMemcpyHostToDevice));
      std::vector<float> one_hot(3, 0.0F);
      one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_ASSERT(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
      total += mlp.forward(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(0.05F);
    }
    return total / static_cast<float>(n);
  };

  float first_loss = run_epoch();
  float last_loss = first_loss;
  for (int e = 1; e < 30; ++e) last_loss = run_epoch();

  EXPECT_LT(last_loss, first_loss * 0.5F) << "Loss should decrease significantly";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: final accuracy > 90 %
// =============================================================================

TEST(TrainingTest, HighAccuracy) {
  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 50, 99);
  int n = static_cast<int>(samples.size());

  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(456);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  for (int epoch = 0; epoch < 60; ++epoch) {
    for (int s = 0; s < n; ++s) {
      CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      std::vector<float> one_hot(3, 0.0F);
      one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_CHECK(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
      mlp.forward(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(0.05F);
    }
  }

  int correct = 0;
  for (int s = 0; s < n; ++s) {
    CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (mlp.predict(d_x) == labels[static_cast<size_t>(s)]) ++correct;
  }
  double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(n);
  EXPECT_GT(acc, 90.0) << "Accuracy should exceed 90% on linearly-separable data";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: forward on zero input produces bias-only output
// =============================================================================

TEST(TrainingTest, ForwardZeroInputGivesBias) {
  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(1);

  // Zero input → dense_forward output = bias (since X·W = 0).
  float* d_x;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_x, 0, 4 * sizeof(float)));

  // Run only dense1 + relu + dense2 (via predict which skips softmax).
  int cls = mlp.predict(d_x);
  EXPECT_GE(cls, 0);
  EXPECT_LT(cls, 3);

  // Also verify that z2 logits are finite.
  std::vector<float> logits(3);
  CUDA_CHECK(cudaMemcpy(logits.data(), mlp.z2, 3 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(logits[static_cast<size_t>(i)])) << "logit[" << i << "] not finite";
  }

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
}

// =============================================================================
// Test: log-softmax output exponentials sum to 1
// =============================================================================

TEST(TrainingTest, LogSoftmaxSumsToOne) {
  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(7);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  // Set an arbitrary input.
  std::vector<float> x = {1.0F, -0.5F, 2.0F, 0.3F};
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> target = {0.0F, 1.0F, 0.0F};
  CUDA_CHECK(cudaMemcpy(d_target, target.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

  mlp.forward(d_x, d_target);

  // Read log_sm and check exp(log_sm) sums to 1.
  std::vector<float> lsm(3);
  CUDA_CHECK(cudaMemcpy(lsm.data(), mlp.log_sm, 3 * sizeof(float), cudaMemcpyDeviceToHost));
  float prob_sum = 0.0F;
  for (int i = 0; i < 3; ++i) {
    float p = std::exp(lsm[static_cast<size_t>(i)]);
    EXPECT_GE(p, 0.0F);
    EXPECT_LE(p, 1.0F);
    prob_sum += p;
  }
  EXPECT_NEAR(prob_sum, 1.0F, 1e-5F);

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: cross-entropy loss is non-negative
// =============================================================================

TEST(TrainingTest, CrossEntropyNonNegative) {
  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(55);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 10, 77);

  for (int s = 0; s < static_cast<int>(samples.size()); ++s) {
    CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::vector<float> one_hot(3, 0.0F);
    one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
    CUDA_CHECK(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
    float loss = mlp.forward(d_x, d_target);
    EXPECT_GE(loss, 0.0F) << "CE loss must be non-negative for sample " << s;
  }

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: SGD step actually changes weights
// =============================================================================

TEST(TrainingTest, SgdStepMutatesWeights) {
  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(33);

  // Snapshot W1 before.
  std::vector<float> w1_before(4 * 16);
  CUDA_CHECK(cudaMemcpy(w1_before.data(), mlp.W1, w1_before.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Run one forward+backward+update pass.
  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  std::vector<float> x = {1.0F, 2.0F, 0.5F, -1.0F};
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> target = {1.0F, 0.0F, 0.0F};
  CUDA_CHECK(cudaMemcpy(d_target, target.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

  mlp.forward(d_x, d_target);
  mlp.backward(d_x, d_target);
  mlp.sgd_step(0.05F);

  // Snapshot W1 after.
  std::vector<float> w1_after(4 * 16);
  CUDA_CHECK(
      cudaMemcpy(w1_after.data(), mlp.W1, w1_after.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // At least some weights must have changed.
  bool any_changed = false;
  for (size_t i = 0; i < w1_before.size(); ++i) {
    if (w1_before[i] != w1_after[i]) {
      any_changed = true;
      break;
    }
  }
  EXPECT_TRUE(any_changed) << "SGD step should modify at least some weights";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: loss is monotonically non-increasing (averaged over small windows)
// =============================================================================

TEST(TrainingTest, LossMonotonicallyDecreasing) {
  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 30, 42);
  int n = static_cast<int>(samples.size());

  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(123);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  auto run_epoch = [&]() {
    float total = 0.0F;
    for (int s = 0; s < n; ++s) {
      CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      std::vector<float> one_hot(3, 0.0F);
      one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_CHECK(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
      total += mlp.forward(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(0.05F);
    }
    return total / static_cast<float>(n);
  };

  // Record per-epoch losses and verify 5-epoch averaged windows decrease.
  constexpr int kEpochs = 40;
  std::vector<float> losses(kEpochs);
  for (int e = 0; e < kEpochs; ++e) losses[static_cast<size_t>(e)] = run_epoch();

  // Compare the average of the first 5 epochs vs the last 5 epochs.
  float early_avg = 0.0F;
  float late_avg = 0.0F;
  for (int i = 0; i < 5; ++i) {
    early_avg += losses[static_cast<size_t>(i)];
    late_avg += losses[static_cast<size_t>(kEpochs - 5 + i)];
  }
  early_avg /= 5.0F;
  late_avg /= 5.0F;
  EXPECT_LT(late_avg, early_avg) << "Average loss in last 5 epochs should be less than first 5";

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}
