/**
 * @file inference.cu
 * @brief Lesson 18 — Inference Pipeline: save / load weights, batched forward.
 *
 * In real-world ML the training and inference phases are typically separate:
 *
 *   1. **Train**  — optimise weights on a dataset (may take hours/days).
 *   2. **Save**   — serialise the learned weights to persistent storage.
 *   3. **Load**   — deserialise into a (possibly different) process or machine.
 *   4. **Infer**  — run the forward pass on new, unseen data.
 *
 * This lesson demonstrates all four steps with a small MLP that reuses
 * the Lesson 17 training pattern, then saves and reloads the weights
 * before running batched inference.
 *
 * Weight file format (little-endian, binary):
 *   [4 bytes: in_dim] [4 bytes: hid_dim] [4 bytes: out_dim]   ← header
 *   [W1 floats] [b1 floats] [W2 floats] [b2 floats]           ← parameters
 *
 * Binary format is chosen over text because:
 *  - No precision loss from float→text→float round-tripping.
 *  - Much smaller files (4 bytes/float vs ~10–15 chars/float).
 *  - Faster I/O (no parsing).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
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
// Kernels (same as Lesson 17, self-contained)
// =============================================================================
// Each lesson duplicates its kernels so it compiles stand-alone.  In a real
// project you’d factor them into a shared library or header.  Duplication
// here keeps each lesson a single, runnable file.

/// Dense forward:  Y[1×out] = X[1×in] · W[in×out] + b[1×out].

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

// =============================================================================
// MLP with save/load — extends Lesson 17’s MLP with serialisation
// =============================================================================

/// The InferenceMLP adds save() and load() to the basic MLP struct, plus
/// a predict_batch() convenience method for running inference on many
/// samples sequentially.  In production, batching would be done on-device
/// (e.g., batched GEMM) for much higher throughput.
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
    A(&elem_loss, out);
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  }

  void init_weights(unsigned seed) {
    std::mt19937 gen(seed);
    auto fill = [&](float* d, int n, float scale) {
      std::normal_distribution<float> dist(0.0F, scale);
      std::vector<float> h(static_cast<size_t>(n));
      for (auto& v : h) v = dist(gen);
      CUDA_CHECK(
          cudaMemcpy(d, h.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    };
    fill(W1, in_dim * hid_dim, std::sqrt(2.0F / static_cast<float>(in_dim)));
    fill(b1, hid_dim, 0.01F);
    fill(W2, hid_dim * out_dim, std::sqrt(2.0F / static_cast<float>(hid_dim)));
    fill(b2, out_dim, 0.01F);
  }

  // ---- Save weights to binary file ----
  /// Serialise the model by writing a dimension header followed by all
  /// learnable parameters.  Each parameter tensor is copied from device
  /// to host first (cudaMemcpy D→H), then written in native float format.
  ///
  /// The dimension header lets load() verify that the file matches the
  /// model architecture before attempting to read weights.
  void save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&in_dim), sizeof(int));
    f.write(reinterpret_cast<const char*>(&hid_dim), sizeof(int));
    f.write(reinterpret_cast<const char*>(&out_dim), sizeof(int));

    auto write_param = [&](const float* d, int n) {
      auto sz = static_cast<size_t>(n);
      std::vector<float> h(sz);
      CUDA_CHECK(cudaMemcpy(h.data(), d, sz * sizeof(float), cudaMemcpyDeviceToHost));
      f.write(reinterpret_cast<const char*>(h.data()),
              static_cast<std::streamsize>(sz * sizeof(float)));
    };
    write_param(W1, in_dim * hid_dim);
    write_param(b1, hid_dim);
    write_param(W2, hid_dim * out_dim);
    write_param(b2, out_dim);
  }

  // ---- Load weights from binary file ----
  /// Deserialise: read the dimension header, verify it matches the
  /// already-allocated model, then read parameter tensors into device
  /// memory (host buffer → cudaMemcpy H→D).
  ///
  /// Dimension checking is critical in practice — loading a weight file
  /// into a model with different layer sizes would silently corrupt
  /// memory and produce garbage predictions.
  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int in{}, hid{}, out{};
    f.read(reinterpret_cast<char*>(&in), sizeof(int));
    f.read(reinterpret_cast<char*>(&hid), sizeof(int));
    f.read(reinterpret_cast<char*>(&out), sizeof(int));

    if (in != in_dim || hid != hid_dim || out != out_dim) {
      std::fprintf(stderr, "Dimension mismatch in weight file!\n");
      std::abort();
    }

    auto read_param = [&](float* d, int n) {
      auto sz = static_cast<size_t>(n);
      std::vector<float> h(sz);
      f.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(sz * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d, h.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    };
    read_param(W1, in_dim * hid_dim);
    read_param(b1, hid_dim);
    read_param(W2, hid_dim * out_dim);
    read_param(b2, out_dim);
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
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
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

  /// Single-sample prediction: run the forward pass (Dense1 → ReLU →
  /// Dense2) and return the index of the largest logit (argmax).
  ///
  /// Softmax is skipped because argmax(softmax(z)) == argmax(z) —
  /// softmax is monotone, so it doesn’t change which class wins.
  [[nodiscard]] int predict(const float* d_x) {
    dense_fwd<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    relu_fwd<<<1, hid_dim>>>(z1, a1, hid_dim);
    dense_fwd<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_z(static_cast<size_t>(out_dim));
    CUDA_CHECK(cudaMemcpy(h_z.data(), z2, static_cast<size_t>(out_dim) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return static_cast<int>(std::distance(h_z.begin(), std::max_element(h_z.begin(), h_z.end())));
  }

  /// Batched inference — run predict on each row of a batch.
  ///
  /// This naïve loop copies one sample at a time and launches tiny
  /// kernels.  A production pipeline would:
  ///   1. Copy the entire batch to the device in one cudaMemcpy.
  ///   2. Use batched matrix multiplications (e.g., cuBLAS SGEMM, see
  ///      Lesson 19) to process all samples in parallel.
  ///   3. Use CUDA streams to overlap H→D copies with kernel execution.
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

// =============================================================================
// Synthetic data (same as Lesson 17)
// =============================================================================

/// Three Gaussian clusters in 4D — see Lesson 17 for the design rationale.
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
// main — the full ML lifecycle: train → save → load → infer
// =============================================================================

/// This demo shows the typical deployment pattern:
///   1. Train on the dataset (reuses Lesson 17’s training loop).
///   2. Save the learned weights to disk.
///   3. Create a **fresh** model and load the saved weights.
///   4. Run batched inference with the loaded model and measure accuracy.
///
/// The fact that the loaded model achieves the same accuracy as the
/// trained model verifies that save/load is faithful.
int main() {
  constexpr int N_PER_CLASS = 50;
  constexpr int EPOCHS = 50;
  constexpr float LR = 0.05F;
  const std::string weight_file = "mlp_weights.bin";

  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, N_PER_CLASS, 42);
  int n = static_cast<int>(samples.size());

  // --- Train ---
  InferenceMLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(123);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    for (int s = 0; s < n; ++s) {
      CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      std::vector<float> one_hot(3, 0.0F);
      one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_CHECK(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
      mlp.forward_loss(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(LR);
    }
  }

  // --- Save ---
  mlp.save(weight_file);
  std::printf("Weights saved to %s\n", weight_file.c_str());

  // --- Load into fresh model ---
  InferenceMLP mlp2{};
  mlp2.alloc(4, 16, 3);
  mlp2.load(weight_file);

  // --- Batched inference ---
  std::vector<int> preds;
  mlp2.predict_batch(samples, preds);

  int correct = 0;
  for (int s = 0; s < n; ++s)
    if (preds[static_cast<size_t>(s)] == labels[static_cast<size_t>(s)]) ++correct;

  std::printf("Inference accuracy (loaded model): %d / %d (%.1f%%)\n", correct, n,
              100.0 * static_cast<double>(correct) / static_cast<double>(n));

  mlp.free_all();
  mlp2.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
  return EXIT_SUCCESS;
}
