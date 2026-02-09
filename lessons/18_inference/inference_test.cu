/**
 * @file inference_test.cu
 * @brief Unit tests for Lesson 18 — Inference Pipeline (save/load, batched).
 *
 * Tests:
 *  1. Save → load round-trip preserves weights exactly.
 *  2. Loaded model predictions match trained model predictions.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) FAIL() << "CUDA error: " << cudaGetErrorString(err_); \
  } while (0)

#define CUDA_ASSERT(call)                                                 \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_)); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

// Kernel definitions (self-contained for single-TU compilation) ——————————————

__global__ void dense_fwd(const float* X, const float* W, const float* b, float* Y, int in_dim,
                          int out_dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < out_dim) {
    float sum = b[j];
    for (int i = 0; i < in_dim; ++i) sum += X[i] * W[i * out_dim + j];
    Y[j] = sum;
  }
}

__global__ void dense_bwd_dX(const float* dY, const float* W, float* dX, int in_dim, int out_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < in_dim) {
    float sum = 0.0F;
    for (int j = 0; j < out_dim; ++j) sum += dY[j] * W[i * out_dim + j];
    dX[i] = sum;
  }
}

__global__ void dense_bwd_dW(const float* X, const float* dY, float* dW, int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / out_dim;
  int j = idx % out_dim;
  if (i < in_dim && j < out_dim) dW[i * out_dim + j] = X[i] * dY[j];
}

__global__ void relu_fwd(const float* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (in[i] > 0.0F) ? in[i] : 0.0F;
}

__global__ void relu_bwd(const float* in, const float* go, float* gi, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) gi[i] = (in[i] > 0.0F) ? go[i] : 0.0F;
}

__global__ void log_softmax_k2(const float* logits, float* log_sm, int N) {
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

__global__ void ce_fwd(const float* log_sm, const float* target, float* elem, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem[i] = -target[i] * log_sm[i];
}

__global__ void ce_bwd(const float* log_sm, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

__global__ void reduce_sum_k2(const float* in, float* out, int N) {
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

__global__ void sgd_upd(float* param, const float* grad, float lr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) param[i] -= lr * grad[i];
}

// InferenceMLP (duplicated for test TU — same as inference.cu)
struct InferenceMLP {
  int in_dim{}, hid_dim{}, out_dim{};
  float *W1{}, *b1{}, *W2{}, *b2{};
  float *dW1{}, *db1{}, *dW2{}, *db2{};
  float *z1{}, *a1{}, *z2{}, *log_sm{};
  float *dz2{}, *da1{}, *dz1{};
  float *elem_loss{}, *d_loss{};

  void alloc(int in, int hid, int out) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    auto A = [](float** p, int n) {
      CUDA_ASSERT(cudaMalloc(p, static_cast<size_t>(n) * sizeof(float)));
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

  void save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&in_dim), sizeof(int));
    f.write(reinterpret_cast<const char*>(&hid_dim), sizeof(int));
    f.write(reinterpret_cast<const char*>(&out_dim), sizeof(int));
    auto wp = [&](const float* d, int n) {
      auto sz = static_cast<size_t>(n);
      std::vector<float> h(sz);
      CUDA_ASSERT(cudaMemcpy(h.data(), d, sz * sizeof(float), cudaMemcpyDeviceToHost));
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(sz * sizeof(float)));
    };
    wp(W1, in_dim * hid_dim);
    wp(b1, hid_dim);
    wp(W2, hid_dim * out_dim);
    wp(b2, out_dim);
  }

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int in{}, hid{}, out{};
    f.read(reinterpret_cast<char*>(&in), sizeof(int));
    f.read(reinterpret_cast<char*>(&hid), sizeof(int));
    f.read(reinterpret_cast<char*>(&out), sizeof(int));
    ASSERT_EQ(in, in_dim);
    ASSERT_EQ(hid, hid_dim);
    ASSERT_EQ(out, out_dim);
    auto rp = [&](float* d, int n) {
      auto sz = static_cast<size_t>(n);
      std::vector<float> h(sz);
      f.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(sz * sizeof(float)));
      CUDA_ASSERT(cudaMemcpy(d, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    };
    rp(W1, in_dim * hid_dim);
    rp(b1, hid_dim);
    rp(W2, hid_dim * out_dim);
    rp(b2, out_dim);
  }

  float forward_loss(const float* d_x, const float* d_target) {
    int bp = 1;
    while (bp < out_dim) bp <<= 1;
    dense_fwd<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    relu_fwd<<<1, hid_dim>>>(z1, a1, hid_dim);
    dense_fwd<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    log_softmax_k2<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(z2, log_sm, out_dim);
    ce_fwd<<<1, out_dim>>>(log_sm, d_target, elem_loss, out_dim);
    reduce_sum_k2<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(elem_loss, d_loss, out_dim);
    CUDA_ASSERT(cudaDeviceSynchronize());
    float h_loss;
    CUDA_ASSERT(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
  }

  void backward(const float* d_x, const float* d_target) {
    ce_bwd<<<1, out_dim>>>(log_sm, d_target, dz2, out_dim);
    dense_bwd_dW<<<1, hid_dim * out_dim>>>(a1, dz2, dW2, hid_dim, out_dim);
    CUDA_CHECK(cudaMemcpy(db2, dz2, static_cast<size_t>(out_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    dense_bwd_dX<<<1, hid_dim>>>(dz2, W2, da1, hid_dim, out_dim);
    relu_bwd<<<1, hid_dim>>>(z1, da1, dz1, hid_dim);
    dense_bwd_dW<<<1, in_dim * hid_dim>>>(d_x, dz1, dW1, in_dim, hid_dim);
    CUDA_CHECK(cudaMemcpy(db1, dz1, static_cast<size_t>(hid_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void sgd_step(float lr) {
    auto upd = [lr](float* p, const float* g, int n) {
      sgd_upd<<<(n + 255) / 256, 256>>>(p, g, lr, n);
    };
    upd(W1, dW1, in_dim * hid_dim);
    upd(b1, db1, hid_dim);
    upd(W2, dW2, hid_dim * out_dim);
    upd(b2, db2, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  [[nodiscard]] int predict(const float* d_x) {
    dense_fwd<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    relu_fwd<<<1, hid_dim>>>(z1, a1, hid_dim);
    dense_fwd<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    CUDA_ASSERT(cudaDeviceSynchronize());
    std::vector<float> h_z(static_cast<size_t>(out_dim));
    CUDA_ASSERT(cudaMemcpy(h_z.data(), z2, static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    return static_cast<int>(std::distance(h_z.begin(), std::max_element(h_z.begin(), h_z.end())));
  }

  void predict_batch(const std::vector<std::vector<float>>& batch, std::vector<int>& preds) {
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(in_dim) * sizeof(float)));
    preds.resize(batch.size());
    for (size_t s = 0; s < batch.size(); ++s) {
      CUDA_CHECK(cudaMemcpy(d_x, batch[s].data(), static_cast<size_t>(in_dim) * sizeof(float),
                            cudaMemcpyHostToDevice));
      preds[s] = predict(d_x);
    }
    CUDA_CHECK(cudaFree(d_x));
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
    cudaFree(elem_loss);
    cudaFree(d_loss);
  }
};

// Helpers ----------

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

static void train_model(InferenceMLP& mlp, const std::vector<std::vector<float>>& samples,
                        const std::vector<int>& labels, int epochs, float lr) {
  int n = static_cast<int>(samples.size());
  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));
  for (int e = 0; e < epochs; ++e) {
    for (int s = 0; s < n; ++s) {
      CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      std::vector<float> oh(3, 0.0F);
      oh[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_CHECK(cudaMemcpy(d_target, oh.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
      mlp.forward_loss(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(lr);
    }
  }
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
}

// =============================================================================
// Test: save → load round-trip preserves weights
// =============================================================================

TEST(InferenceTest, SaveLoadRoundtrip) {
  const std::string path = "/tmp/test_weights_roundtrip.bin";

  InferenceMLP m1{};
  m1.alloc(4, 16, 3);
  m1.init_weights(77);
  m1.save(path);

  InferenceMLP m2{};
  m2.alloc(4, 16, 3);
  m2.load(path);

  // Compare all parameters
  auto compare = [](const float* d1, const float* d2, size_t n, const char* name) {
    std::vector<float> h1(n), h2(n);
    CUDA_CHECK(cudaMemcpy(h1.data(), d1, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d2, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) EXPECT_FLOAT_EQ(h1[i], h2[i]) << name << "[" << i << "]";
  };
  compare(m1.W1, m2.W1, 4 * 16, "W1");
  compare(m1.b1, m2.b1, 16, "b1");
  compare(m1.W2, m2.W2, 16 * 3, "W2");
  compare(m1.b2, m2.b2, 3, "b2");

  m1.free_all();
  m2.free_all();
  std::remove(path.c_str());
}

// =============================================================================
// Test: loaded model produces same predictions as trained model
// =============================================================================

TEST(InferenceTest, LoadedModelMatchesTrained) {
  const std::string path = "/tmp/test_weights_match.bin";

  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 30, 42);

  InferenceMLP trained{};
  trained.alloc(4, 16, 3);
  trained.init_weights(123);
  train_model(trained, samples, labels, 30, 0.05F);
  trained.save(path);

  // Predictions from trained model
  std::vector<int> preds_orig;
  trained.predict_batch(samples, preds_orig);

  // Load into fresh model
  InferenceMLP loaded{};
  loaded.alloc(4, 16, 3);
  loaded.load(path);

  std::vector<int> preds_loaded;
  loaded.predict_batch(samples, preds_loaded);

  ASSERT_EQ(preds_orig.size(), preds_loaded.size());
  for (size_t i = 0; i < preds_orig.size(); ++i)
    EXPECT_EQ(preds_orig[i], preds_loaded[i]) << "Mismatch at sample " << i;

  trained.free_all();
  loaded.free_all();
  std::remove(path.c_str());
}

// =============================================================================
// Test: batched inference accuracy > 90%
// =============================================================================

TEST(InferenceTest, BatchAccuracy) {
  const std::string path = "/tmp/test_weights_acc.bin";

  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, 50, 99);

  InferenceMLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(456);
  train_model(mlp, samples, labels, 60, 0.05F);
  mlp.save(path);

  InferenceMLP loaded{};
  loaded.alloc(4, 16, 3);
  loaded.load(path);

  std::vector<int> preds;
  loaded.predict_batch(samples, preds);

  int correct = 0;
  int n = static_cast<int>(samples.size());
  for (int s = 0; s < n; ++s)
    if (preds[static_cast<size_t>(s)] == labels[static_cast<size_t>(s)]) ++correct;

  double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(n);
  EXPECT_GT(acc, 90.0) << "Loaded model accuracy should exceed 90%";

  mlp.free_all();
  loaded.free_all();
  std::remove(path.c_str());
}
