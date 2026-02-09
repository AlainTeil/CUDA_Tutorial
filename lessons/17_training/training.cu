/**
 * @file training.cu
 * @brief Lesson 17 — End-to-End Training Loop (MLP on synthetic data).
 *
 * Builds a simple 2-layer MLP:  Input(4) → Dense(16) → ReLU → Dense(3) → Softmax+CE
 * Trains on synthetically-generated linearly-separable data using SGD.
 *
 * This lesson wires together the building blocks from Lessons 12–16 into
 * the fundamental deep-learning cycle:
 *
 *   1. **Forward pass**  — compute predictions from input.
 *   2. **Loss**          — measure prediction quality (scalar).
 *   3. **Backward pass** — compute gradients w.r.t. every parameter
 *                          by applying the **chain rule** in reverse
 *                          through the computational graph.
 *   4. **Parameter update** — nudge weights opposite to the gradient
 *                             (SGD: w -= lr * dw).
 *
 * Key design choices in this demo:
 *  - **Online (single-sample) SGD**: each sample triggers its own
 *    forward→backward→update.  This is noisy but simple; real training
 *    uses mini-batches for better GPU utilisation and smoother gradients.
 *  - **He initialisation**: weight std = sqrt(2/fan_in), which keeps
 *    activations and gradients in a healthy range when using ReLU.
 *  - **One-hot targets**: the class label is encoded as a vector with
 *    1 at the true class and 0 elsewhere, matching cross-entropy’s
 *    expectation of a probability distribution.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// =============================================================================
// Kernels (self-contained for this lesson)
// =============================================================================

/// Dense (fully-connected) forward:  Y = X · W + b
///
/// For a single sample:  X is [1×in_dim], W is [in_dim×out_dim], Y is [1×out_dim].
/// Each thread computes one output neuron by dot-producting X with a column
/// of W and adding the bias.  This is essentially a matrix-vector multiply.
__global__ void dense_forward(const float* X, const float* W, const float* b, float* Y, int in_dim,
                              int out_dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron index
  if (j < out_dim) {
    float sum = b[j];
    for (int i = 0; i < in_dim; ++i) sum += X[i] * W[i * out_dim + j];
    Y[j] = sum;
  }
}

/// Dense backward — gradient w.r.t. input X:
///   dX = dY · W^T
/// Each thread computes one element of dX by dot-producting dY with a row of W.
/// (A row of W corresponds to all the weights leaving input i.)
__global__ void dense_backward_dX(const float* dY, const float* W, float* dX, int in_dim,
                                  int out_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // input neuron index
  if (i < in_dim) {
    float sum = 0.0F;
    for (int j = 0; j < out_dim; ++j) sum += dY[j] * W[i * out_dim + j];
    dX[i] = sum;
  }
}

/// Dense backward — gradient w.r.t. weights W:
///   dW = X^T · dY   (outer product for a single sample)
/// Each thread handles one element dW[i][j] = X[i] * dY[j].
__global__ void dense_backward_dW(const float* X, const float* dY, float* dW, int in_dim,
                                  int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / out_dim;
  int j = idx % out_dim;
  if (i < in_dim && j < out_dim) {
    dW[i * out_dim + j] = X[i] * dY[j];
  }
}

/// ReLU forward: out = max(0, in).
/// ReLU is the most common activation because it is cheap to compute,
/// provides non-linearity, and avoids the vanishing-gradient problem
/// that plagues sigmoid/tanh in deep networks.
__global__ void relu_forward(const float* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (in[i] > 0.0F) ? in[i] : 0.0F;
}

/// ReLU backward: gradient passes through unchanged where the input was
/// positive, and is zeroed where the input was ≤ 0 (“dead zone”).
__global__ void relu_backward(const float* in, const float* grad_out, float* grad_in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad_in[i] = (in[i] > 0.0F) ? grad_out[i] : 0.0F;
}

/// Stable log-softmax (see Lesson 16 for the full derivation).
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

/// CE forward: per-element loss = -target_i * log_softmax_i
__global__ void ce_forward_k(const float* log_sm, const float* target, float* elem, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem[i] = -target[i] * log_sm[i];
}

/// CE backward: the combined softmax+CE gradient = softmax_i - target_i.
/// (See Lesson 16 for the derivation.)
__global__ void ce_backward_k(const float* log_sm, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

/// Reduce sum (see Lesson 16 for the shared-memory reduction pattern).
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

/// Vanilla SGD: param -= lr * grad.
/// The simplest optimiser — each parameter is updated independently.
/// Modern training uses variants like Adam that maintain per-parameter
/// running statistics, but SGD remains a solid baseline.
__global__ void sgd_update(float* param, const float* grad, float lr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) param[i] -= lr * grad[i];
}

// =============================================================================
// MLP struct — manages the computational graph for a two-layer network
// =============================================================================

/// Encapsulates device buffers for weights, activations, and gradients.
///
/// The computational graph for a single forward→backward is:
///
///   x → [Dense1] → z1 → [ReLU] → a1 → [Dense2] → z2 → [Softmax+CE] → loss
///
/// Backward reverses this chain, computing gradients at each node and
/// accumulating dW, db for each Dense layer.
struct MLP {
  int in_dim, hid_dim, out_dim;

  // Weights/biases on device — the *learnable* parameters.
  float *W1, *b1, *W2, *b2;
  // Gradient buffers — dW/db hold ∂L/∂W and ∂L/∂b after backward().
  float *dW1, *db1, *dW2, *db2;
  // Intermediate activations — needed by backward to compute gradients.
  float *z1, *a1, *z2, *log_sm;
  // Gradient intermediaries flowing backward through the graph.
  float *dz2, *da1, *dz1, *dx;
  // Loss buffers (per-element and scalar).
  float *elem_loss, *d_loss;

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

  /// He (Kaiming) initialisation:  std = sqrt(2 / fan_in).
  ///
  /// With ReLU, roughly half the neurons output zero, so the “effective
  /// fan-in” is halved — the factor of 2 compensates for that.  Without
  /// this, deep networks suffer from vanishing or exploding activations
  /// in early training.
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

  /// Forward pass: x → Dense1 → ReLU → Dense2 → LogSoftmax → CE → loss.
  ///
  /// Each kernel is launched with 1 block because we process a single
  /// sample at a time (online SGD).  In production you would batch
  /// samples to fill the GPU’s thousands of cores.
  float forward(const float* d_x, const float* d_target) {
    // Block size for shared-memory reductions must be a power of two.
    int bp = 1;
    while (bp < out_dim) bp <<= 1;

    dense_forward<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    relu_forward<<<1, hid_dim>>>(z1, a1, hid_dim);
    dense_forward<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    log_softmax_k<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(z2, log_sm, out_dim);
    ce_forward_k<<<1, out_dim>>>(log_sm, d_target, elem_loss, out_dim);
    reduce_sum_k<<<1, bp, static_cast<size_t>(bp) * sizeof(float)>>>(elem_loss, d_loss, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
  }

  /// Backward pass — reverse the computational graph.
  ///
  /// The chain rule dictates the order: the gradient of the loss w.r.t.
  /// a parameter deep in the network flows *backward* through every
  /// intervening operation.  We therefore process layers from output to
  /// input.
  void backward(const float* d_x, const float* d_target) {
    // ------ Output layer (Dense2) ------
    // Combined softmax+CE gradient:  dz2 = softmax(z2) - target
    ce_backward_k<<<1, out_dim>>>(log_sm, d_target, dz2, out_dim);

    // Weight gradient: dW2 = a1^T · dz2  (outer product, single sample)
    dense_backward_dW<<<1, hid_dim * out_dim>>>(a1, dz2, dW2, hid_dim, out_dim);
    // Bias gradient: db2 = dz2  (just copy — bias adds 1·dz2)
    CUDA_CHECK(cudaMemcpy(db2, dz2, static_cast<size_t>(out_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // ------ Propagate through Dense2 to hidden layer ------
    // da1 = dz2 · W2^T  (how much each hidden neuron contributed to the error)
    dense_backward_dX<<<1, hid_dim>>>(dz2, W2, da1, hid_dim, out_dim);

    // ------ ReLU backward: zero out gradients where activation was ≤ 0 ------
    relu_backward<<<1, hid_dim>>>(z1, da1, dz1, hid_dim);

    // ------ Hidden layer (Dense1) ------
    dense_backward_dW<<<1, in_dim * hid_dim>>>(d_x, dz1, dW1, in_dim, hid_dim);
    CUDA_CHECK(cudaMemcpy(db1, dz1, static_cast<size_t>(hid_dim) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  /// SGD update: w -= lr * dw  for every learnable parameter.
  ///
  /// We use ceil-division  (n + 255) / 256  to launch enough blocks
  /// so that every element gets a thread, even when N is not a multiple
  /// of 256.  Each thread updates one parameter independently.
  void sgd_step(float lr) {
    auto update = [lr](float* p, const float* g, int n) {
      sgd_update<<<(n + 255) / 256, 256>>>(p, g, lr, n);
    };
    update(W1, dW1, in_dim * hid_dim);
    update(b1, db1, hid_dim);
    update(W2, dW2, hid_dim * out_dim);
    update(b2, db2, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  /// Predict the class of a single input by running a forward pass
  /// through Dense1 → ReLU → Dense2 and returning the argmax of the
  /// output logits.  Note: softmax is not needed for prediction because
  /// it is monotonic — the largest logit always maps to the largest
  /// probability.
  [[nodiscard]] int predict(const float* d_x) {
    int bp = 1;
    while (bp < out_dim) bp <<= 1;
    dense_forward<<<1, hid_dim>>>(d_x, W1, b1, z1, in_dim, hid_dim);
    relu_forward<<<1, hid_dim>>>(z1, a1, hid_dim);
    dense_forward<<<1, out_dim>>>(a1, W2, b2, z2, hid_dim, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_z(static_cast<size_t>(out_dim));
    CUDA_CHECK(cudaMemcpy(h_z.data(), z2, static_cast<size_t>(out_dim) * sizeof(float),
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

// =============================================================================
// Synthetic data: 3 classes, linearly separable in 4D
// =============================================================================

/// Generates clusters centred at (2,0,0,0), (0,2,0,0), (0,0,2,0) with
/// Gaussian noise (std = 0.3).  These clusters are well separated, so
/// even a simple MLP should reach near-perfect accuracy — making this a
/// good sanity-check dataset that lets us focus on the training mechanics
/// rather than data engineering.
void generate_data(std::vector<std::vector<float>>& samples, std::vector<int>& labels,
                   int n_per_class, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  // Class centres in 4D
  float centres[3][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      std::vector<float> s(4);
      for (int d = 0; d < 4; ++d) s[static_cast<size_t>(d)] = centres[c][d] + noise(gen);
      samples.push_back(s);
      labels.push_back(c);
    }
  }
}

// =============================================================================
// main — the training loop
// =============================================================================

/// Demonstrates the complete training cycle: data generation → weight init
/// → (forward → backward → SGD update) × epochs → evaluation.
int main() {
  constexpr int N_PER_CLASS = 50;
  constexpr int EPOCHS = 50;
  constexpr float LR = 0.05F;

  std::vector<std::vector<float>> samples;
  std::vector<int> labels;
  generate_data(samples, labels, N_PER_CLASS, 42);

  MLP mlp{};
  mlp.alloc(4, 16, 3);
  mlp.init_weights(123);

  float* d_x;
  float* d_target;
  CUDA_CHECK(cudaMalloc(&d_x, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_target, 3 * sizeof(float)));

  int n = static_cast<int>(samples.size());

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    float total_loss = 0.0F;
    for (int s = 0; s < n; ++s) {
      // Copy one sample to device.
      CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      // Encode the integer label as a one-hot vector so that cross-entropy
      // can compute -Σ target_i * log(softmax_i).
      std::vector<float> one_hot(3, 0.0F);
      one_hot[static_cast<size_t>(labels[static_cast<size_t>(s)])] = 1.0F;
      CUDA_CHECK(cudaMemcpy(d_target, one_hot.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

      // The core training cycle: forward → backward → update.
      float loss = mlp.forward(d_x, d_target);
      mlp.backward(d_x, d_target);
      mlp.sgd_step(LR);
      total_loss += loss;
    }
    if (epoch % 10 == 0)
      std::printf("Epoch %3d  avg loss: %.4f\n", epoch,
                  static_cast<double>(total_loss / static_cast<float>(n)));
  }

  // Accuracy
  int correct = 0;
  for (int s = 0; s < n; ++s) {
    CUDA_CHECK(cudaMemcpy(d_x, samples[static_cast<size_t>(s)].data(), 4 * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (mlp.predict(d_x) == labels[static_cast<size_t>(s)]) ++correct;
  }
  std::printf("Train accuracy: %d / %d (%.1f%%)\n", correct, n,
              100.0 * static_cast<double>(correct) / static_cast<double>(n));

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_target));
  return EXIT_SUCCESS;
}
