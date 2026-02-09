/**
 * @file minibatch_training.cu
 * @brief Lesson 20 — Mini-Batch Training with cuBLAS GEMM.
 *
 * Lesson 17 trained with **online SGD** (one sample at a time).  That is
 * simple but wastes the GPU's massive parallelism: each kernel launch
 * processes just a few dozen floats while the GPU has thousands of cores
 * sitting idle.
 *
 * This lesson upgrades the training loop to **mini-batch SGD**, which:
 *
 *   1. **Packs B samples** into a matrix X[B×in_dim] and processes
 *      the entire batch with a single matrix multiply per layer.
 *   2. **Uses cuBLAS SGEMM** for the matrix multiplies — giving us
 *      highly optimised, Tensor-Core-enabled performance for free.
 *   3. **Averages gradients** over the batch before updating weights,
 *      producing smoother gradient estimates than online SGD.
 *
 * The architecture is the same 2-layer MLP:
 *   X[B×4] → Dense(16) → ReLU → Dense(3) → Softmax+CE → loss
 *
 * Key differences from Lesson 17:
 *
 * | Aspect              | Lesson 17 (online)         | Lesson 20 (mini-batch)     |
 * |---------------------|----------------------------|----------------------------|
 * | Batch size          | 1                          | B (configurable)           |
 * | Dense forward       | Hand-written 1-D kernel    | cuBLAS SGEMM               |
 * | Dense backward dW   | Outer-product kernel       | cuBLAS SGEMM (X^T · dY)   |
 * | Dense backward dX   | 1-D kernel                 | cuBLAS SGEMM (dY · W^T)   |
 * | Gradient averaging  | N/A                        | Explicit 1/B scale         |
 * | Bias gradient       | Copy dY (B=1)              | Column-sum reduction       |
 * | GPU utilisation      | Very low                   | Much higher                |
 *
 * Build: this lesson requires cuBLAS (always available with the CUDA Toolkit).
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// =============================================================================
// Error-checking macros
// =============================================================================

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                          \
  do {                                                                              \
    cublasStatus_t st_ = (call);                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                             \
      std::fprintf(stderr, "cuBLAS error at %s:%d — code %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                          \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

// =============================================================================
// cuBLAS row-major helper
// =============================================================================

/// Compute C[M×N] = alpha * A[M×K] * B[K×N] + beta * C[M×N]  (row-major).
///
/// cuBLAS uses column-major, so we apply the classic transpose trick:
///   C_rm = A_rm · B_rm   ←→   C^T_cm = B^T_cm · A^T_cm
/// Since row-major layout == column-major transpose, we simply swap A↔B
/// and M↔N in the cublasSgemm call.  (See Lesson 19 for the derivation.)
static void gemm_rm(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                    const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/// C[M×N] = alpha * A^T[M×K] * B[K×N] + beta * C   (A stored as [K×M] row-major).
/// Transposing A in row-major is equivalent to no-transpose in column-major
/// for the swapped call.  We use CUBLAS_OP_T on the "swapped-B" (= our A).
static void gemm_rm_AT(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  // In the swapped world: C^T = B^T · (A^T)^T = B^T · A_cm
  // "A_cm" = A row-major (K×M) = column-major M×K, so op = CUBLAS_OP_N on it.
  // Wait — let's think carefully:
  //  We want: C[M×N] = A^T[M×K] · B[K×N]   (row-major)
  //  Swap trick: C^T_cm[N×M] = B^T_cm[N×K] · (A^T)^T_cm[K×M]
  //  A is stored [K×M] row-major = [M×K] col-major.  (A^T)^T_cm = A_cm = [M×K].
  //  So we need OP_N on A_cm (lda = M) and OP_N on B (ldb = N).
  //  cublasSgemm(h, OP_N, OP_N, M, N, K, ..., A, M, B, N, ..., C, M) — NO,
  //  that gives C_cm [M×N], but we want C_rm [M×N] = C_cm^T [N×M].
  //  We need: cublasSgemm(h, OP_N, OP_T, N, M, K, ..., B, N, A, M, ..., C, N).
  //  OP_T on "A" in the cuBLAS call means transpose the [M×K] col-major → [K×M],
  //  which is what we want for the right side in  C^T = B · A^T.
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

// =============================================================================
// Element-wise CUDA kernels
// =============================================================================

/// ReLU forward: out[i] = max(0, in[i]).  Works on any contiguous buffer.
__global__ void relu_fwd(const float* in, float* out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = (in[i] > 0.0F) ? in[i] : 0.0F;
}

/// ReLU backward: pass gradient where activation > 0, zero elsewhere.
__global__ void relu_bwd(const float* pre_act, const float* grad_out, float* grad_in, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad_in[i] = (pre_act[i] > 0.0F) ? grad_out[i] : 0.0F;
}

/// Add bias to every row of a [B×D] matrix: Y[b][j] += bias[j].
/// Each thread handles one element of the B×D matrix.
__global__ void add_bias(float* Y, const float* bias, int B, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * D) {
    int j = idx % D;
    Y[idx] += bias[j];
  }
}

/// Per-row log-softmax for a [B×C] matrix of logits.
///
/// Each row (sample) gets its own block.  Within the block, threads
/// collaborate on the parallel reduction (max, then sum-of-exp).
/// blockDim.x must be a power-of-two ≥ C.
__global__ void batch_log_softmax(const float* logits, float* log_sm, int C) {
  extern __shared__ float sdata[];
  int b = blockIdx.x;  // sample index within the batch
  int tid = threadIdx.x;
  const float* row = logits + b * C;

  // Load into shared; out-of-range threads get a sentinel.
  float val = (tid < C) ? row[tid] : -1e30F;
  sdata[tid] = val;
  __syncthreads();

  // Parallel max reduction.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float mx = sdata[0];
  __syncthreads();

  // exp(val − max) and parallel sum.
  float ev = (tid < C) ? expf(val - mx) : 0.0F;
  sdata[tid] = ev;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float lse = logf(sdata[0]);

  if (tid < C) log_sm[b * C + tid] = (val - mx) - lse;
}

/// Cross-entropy forward (batched): elem[b][i] = −target[b][i] · log_sm[b][i].
/// The per-sample loss is the row-sum of elem.
__global__ void ce_fwd(const float* log_sm, const float* target, float* elem, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) elem[i] = -target[i] * log_sm[i];
}

/// Cross-entropy backward (batched):  grad[i] = softmax[i] − target[i].
/// Combined softmax+CE gradient, applied element-wise to the B×C matrix.
__global__ void ce_bwd(const float* log_sm, const float* target, float* grad, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) grad[i] = expf(log_sm[i]) - target[i];
}

/// Reduce sum of a B×C element-loss matrix into a single scalar.
/// A simple single-block reduction (fine for small B×C).
__global__ void reduce_sum_all(const float* in, float* out, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  // Grid-stride accumulation for inputs larger than blockDim.
  float acc = 0.0F;
  for (int i = tid; i < N; i += blockDim.x) acc += in[i];
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) *out = sdata[0];
}

/// Sum columns of a [B×D] matrix → a vector of length D.
/// Computes:  out[j] = Σ_b  in[b*D + j]   for each j.
///
/// Used to compute bias gradients: db = Σ_b dY[b,:].
/// Each thread handles one column j and loops over rows.
__global__ void col_sum(const float* in, float* out, int B, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) {
    float s = 0.0F;
    for (int b = 0; b < B; ++b) s += in[b * D + j];
    out[j] = s;
  }
}

/// Scale every element by a constant:  x[i] *= scale.
/// Used to average gradients over the batch: scale = 1/B.
__global__ void scale_kernel(float* x, float scale, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] *= scale;
}

/// SGD update: param[i] -= lr * grad[i].
__global__ void sgd_update(float* param, const float* grad, float lr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) param[i] -= lr * grad[i];
}

// =============================================================================
// BatchMLP — mini-batch MLP using cuBLAS for dense layers
// =============================================================================

/// A 2-layer MLP that processes B samples simultaneously.
///
/// Buffers are allocated for the maximum batch size up front.
/// The computational graph for a batch:
///
///   X[B×4] →─[GEMM]→ Z1[B×16] + bias1 →─[ReLU]→ A1[B×16]
///          →─[GEMM]→ Z2[B×3]  + bias2 →─[LogSoftmax]→ LS[B×3]
///          →─[CE]──→ loss (scalar, averaged over B)
///
/// The backward pass reverses this, producing dW1, db1, dW2, db2.
struct BatchMLP {
  int in_dim{}, hid_dim{}, out_dim{}, max_batch{};
  cublasHandle_t cublas{};

  // ---------- Learnable parameters ----------
  float *W1{}, *b1{}, *W2{}, *b2{};

  // ---------- Gradient accumulators ----------
  float *dW1{}, *db1{}, *dW2{}, *db2{};

  // ---------- Activation / intermediate buffers (sized for max_batch) ----------
  float *Z1{}, *A1{}, *Z2{}, *LS{};  // forward
  float *dZ2{}, *dA1{}, *dZ1{};      // backward
  float *elem_loss{}, *d_loss{};     // loss

  void alloc(int in, int hid, int out, int max_B) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    max_batch = max_B;
    CUBLAS_CHECK(cublasCreate(&cublas));

    auto A = [](float** p, int n) {
      CUDA_CHECK(cudaMalloc(p, static_cast<size_t>(n) * sizeof(float)));
    };

    // Parameters (batch-size independent).
    A(&W1, in * hid);
    A(&b1, hid);
    A(&W2, hid * out);
    A(&b2, out);

    // Gradients for parameters.
    A(&dW1, in * hid);
    A(&db1, hid);
    A(&dW2, hid * out);
    A(&db2, out);

    // Activation buffers — allocate for max batch size.
    A(&Z1, max_B * hid);
    A(&A1, max_B * hid);
    A(&Z2, max_B * out);
    A(&LS, max_B * out);

    // Backward intermediaries.
    A(&dZ2, max_B * out);
    A(&dA1, max_B * hid);
    A(&dZ1, max_B * hid);

    // Loss buffers.
    A(&elem_loss, max_B * out);
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  }

  /// He initialisation — identical to Lesson 17.
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

  // ------------------------------------------------------------------
  // Forward pass  (batched)
  // ------------------------------------------------------------------

  /// Forward:  X[B×in] → Z1=X·W1+b1 → A1=relu(Z1) → Z2=A1·W2+b2
  ///           → LS=log_softmax(Z2) → CE → loss (scalar).
  ///
  /// The key insight: Dense forward for a batch is a matrix multiply
  ///   Z1[B×hid] = X[B×in] · W1[in×hid]
  /// which is exactly what cuBLAS SGEMM computes in one highly optimised
  /// call — no per-sample loops needed.
  float forward(const float* d_X, const float* d_target, int B) {
    int total1 = B * hid_dim;
    int total2 = B * out_dim;
    int blk = 256;

    // --- Layer 1: Z1 = X · W1 ---
    // cuBLAS SGEMM replaces our hand-written dense_forward kernel.
    // One call processes all B samples simultaneously.
    gemm_rm(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(total1 + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);

    // --- ReLU ---
    relu_fwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, A1, total1);

    // --- Layer 2: Z2 = A1 · W2 ---
    gemm_rm(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(total2 + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);

    // --- Log-softmax (per-row) ---
    int sm_block = 1;
    while (sm_block < out_dim) sm_block <<= 1;
    batch_log_softmax<<<B, sm_block, static_cast<size_t>(sm_block) * sizeof(float)>>>(Z2, LS,
                                                                                      out_dim);

    // --- Cross-entropy loss ---
    ce_fwd<<<(total2 + blk - 1) / blk, blk>>>(LS, d_target, elem_loss, total2);

    // Sum all B×C element losses into a single scalar, then average.
    int red_block = 256;
    reduce_sum_all<<<1, red_block, static_cast<size_t>(red_block) * sizeof(float)>>>(
        elem_loss, d_loss, total2);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss / static_cast<float>(B);  // average loss per sample
  }

  // ------------------------------------------------------------------
  // Backward pass  (batched)
  // ------------------------------------------------------------------

  /// Backward:  compute dW1, db1, dW2, db2 averaged over the batch.
  ///
  /// The chain rule for batched dense layers becomes:
  ///   dZ2[B×out] = softmax − target           (element-wise)
  ///   dW2[hid×out] = (1/B) · A1^T[hid×B] · dZ2[B×out]    ← cuBLAS GEMM
  ///   db2[out]     = (1/B) · Σ_b dZ2[b,:]                 ← column sum
  ///   dA1[B×hid]   = dZ2[B×out] · W2^T[out×hid]           ← cuBLAS GEMM
  ///   dZ1[B×hid]   = dA1 ⊙ relu'(Z1)                     (element-wise)
  ///   dW1[in×hid]  = (1/B) · X^T[in×B] · dZ1[B×hid]      ← cuBLAS GEMM
  ///   db1[hid]     = (1/B) · Σ_b dZ1[b,:]                 ← column sum
  ///
  /// Notice: each dW computation is a matrix multiply (not an outer product),
  /// because we are summing contributions from all B samples at once.
  void backward(const float* d_X, const float* d_target, int B) {
    int total1 = B * hid_dim;
    int total2 = B * out_dim;
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);

    // --- dZ2 = softmax(Z2) − target  [B×out] ---
    ce_bwd<<<(total2 + blk - 1) / blk, blk>>>(LS, d_target, dZ2, total2);

    // --- dW2 = (1/B) · A1^T · dZ2  [hid×out] ---
    // A1 is stored as [B×hid] row-major.  We need A1^T[hid×B] · dZ2[B×out].
    // gemm_rm_AT handles the "transpose first operand" case.
    gemm_rm_AT(cublas, hid_dim, B, out_dim, inv_B, A1, dZ2, 0.0F, dW2);

    // --- db2 = (1/B) · col_sum(dZ2)  [out] ---
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);

    // --- dA1 = dZ2 · W2^T  [B×hid] ---
    // We need  dA1[B×hid] = dZ2[B×out] · W2^T[out×hid].
    // Column-major view of the stored buffers:
    //   W2  rm[hid×out]  →  W2  cm[out×hid]
    //   dZ2 rm[B×out]    →  dZ2 cm[out×B]
    //   dA1 rm[B×hid]    →  dA1 cm[hid×B]
    // We need: dA1_cm[hid×B] = W2^T_cm[hid×out] × dZ2_cm[out×B].
    //   → transa = OP_T on W2_cm[out×hid] → [hid×out],  lda = out_dim
    //   → transb = OP_N on dZ2_cm[out×B]  → [out×B],    ldb = out_dim
    //   → m = hid_dim, n = B, k = out_dim
    {
      float one = 1.0F, zero = 0.0F;
      CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2,
                               out_dim, dZ2, out_dim, &zero, dA1, hid_dim));
    }

    // --- dZ1 = dA1 ⊙ relu'(Z1)  [B×hid] ---
    relu_bwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, total1);

    // --- dW1 = (1/B) · X^T · dZ1  [in×hid] ---
    gemm_rm_AT(cublas, in_dim, B, hid_dim, inv_B, d_X, dZ1, 0.0F, dW1);

    // --- db1 = (1/B) · col_sum(dZ1)  [hid] ---
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ------------------------------------------------------------------
  // SGD update
  // ------------------------------------------------------------------

  void sgd_step(float lr) {
    int blk = 256;
    auto upd = [&](float* p, const float* g, int n) {
      sgd_update<<<(n + blk - 1) / blk, blk>>>(p, g, lr, n);
    };
    upd(W1, dW1, in_dim * hid_dim);
    upd(b1, db1, hid_dim);
    upd(W2, dW2, hid_dim * out_dim);
    upd(b2, db2, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ------------------------------------------------------------------
  // Prediction
  // ------------------------------------------------------------------

  /// Batched prediction: run forward (Dense1→ReLU→Dense2) and return
  /// the argmax class for each sample.
  void predict_batch(const float* d_X, int B, std::vector<int>& preds) {
    int blk = 256;
    int total1 = B * hid_dim;
    int total2 = B * out_dim;

    gemm_rm(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(total1 + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(total1 + blk - 1) / blk, blk>>>(Z1, A1, total1);
    gemm_rm(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(total2 + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy logits to host and argmax each row.
    std::vector<float> h_z(static_cast<size_t>(total2));
    CUDA_CHECK(cudaMemcpy(h_z.data(), Z2, static_cast<size_t>(total2) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    preds.resize(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
      auto row = h_z.begin() + b * out_dim;
      preds[static_cast<size_t>(b)] =
          static_cast<int>(std::distance(row, std::max_element(row, row + out_dim)));
    }
  }

  // ------------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------------

  void free_all() {
    cublasDestroy(cublas);
    for (float* p :
         {W1, b1, W2, b2, dW1, db1, dW2, db2, Z1, A1, Z2, LS, dZ2, dA1, dZ1, elem_loss, d_loss})
      cudaFree(p);
  }
};

// =============================================================================
// Synthetic data (same clusters as Lesson 17)
// =============================================================================

/// Generate 3 Gaussian clusters in 4D — see Lesson 17.
static void generate_data(std::vector<float>& flat_samples, std::vector<int>& labels,
                          int n_per_class, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  float centres[3][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
  int N = 3 * n_per_class;
  flat_samples.resize(static_cast<size_t>(N) * 4);
  labels.resize(static_cast<size_t>(N));
  int idx = 0;
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < n_per_class; ++i) {
      for (int d = 0; d < 4; ++d)
        flat_samples[static_cast<size_t>(idx) * 4 + d] = centres[c][d] + noise(gen);
      labels[static_cast<size_t>(idx)] = c;
      ++idx;
    }
}

/// Build a one-hot target matrix [N×C] from integer labels.
static std::vector<float> make_one_hot(const std::vector<int>& labels, int C) {
  auto N = static_cast<int>(labels.size());
  std::vector<float> oh(static_cast<size_t>(N) * C, 0.0F);
  for (int i = 0; i < N; ++i)
    oh[static_cast<size_t>(i) * C + labels[static_cast<size_t>(i)]] = 1.0F;
  return oh;
}

// =============================================================================
// main — mini-batch training demo
// =============================================================================

int main() {
  constexpr int N_PER_CLASS = 50;  // samples per class
  constexpr int N_TOTAL = 3 * N_PER_CLASS;
  constexpr int BATCH_SIZE = 32;  // the defining feature of this lesson
  constexpr int EPOCHS = 60;
  constexpr float LR = 0.1F;  // can be larger with mini-batch SGD

  // --- Generate data ---
  std::vector<float> flat_data;
  std::vector<int> labels;
  generate_data(flat_data, labels, N_PER_CLASS, 42);
  std::vector<float> one_hot = make_one_hot(labels, 3);

  // --- Allocate model ---
  BatchMLP mlp{};
  mlp.alloc(4, 16, 3, BATCH_SIZE);
  mlp.init_weights(123);

  // --- Copy full dataset to device ---
  // Unlike Lesson 17, we copy the entire dataset once and slice batches
  // from device memory — avoiding per-sample H→D transfers.
  float* d_data;
  float* d_targets;
  CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(N_TOTAL) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_targets, static_cast<size_t>(N_TOTAL) * 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, flat_data.data(), static_cast<size_t>(N_TOTAL) * 4 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_targets, one_hot.data(), static_cast<size_t>(N_TOTAL) * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Shuffle indices ---
  std::vector<int> indices(N_TOTAL);
  std::iota(indices.begin(), indices.end(), 0);

  // Temporary device buffers for the current batch.
  float* d_batch_x;
  float* d_batch_t;
  CUDA_CHECK(cudaMalloc(&d_batch_x, static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_batch_t, static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float)));

  // --- Training loop ---
  std::mt19937 rng(7);
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);
    float epoch_loss = 0.0F;
    int n_batches = 0;

    for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
      // Gather the batch from the shuffled indices.
      // In production you'd use a contiguous permutation buffer on device;
      // here we gather on host for clarity.
      std::vector<float> batch_x(static_cast<size_t>(BATCH_SIZE) * 4);
      std::vector<float> batch_t(static_cast<size_t>(BATCH_SIZE) * 3);
      for (int b = 0; b < BATCH_SIZE; ++b) {
        int si = indices[static_cast<size_t>(start + b)];
        std::memcpy(&batch_x[static_cast<size_t>(b) * 4], &flat_data[static_cast<size_t>(si) * 4],
                    4 * sizeof(float));
        std::memcpy(&batch_t[static_cast<size_t>(b) * 3], &one_hot[static_cast<size_t>(si) * 3],
                    3 * sizeof(float));
      }
      CUDA_CHECK(cudaMemcpy(d_batch_x, batch_x.data(),
                            static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_batch_t, batch_t.data(),
                            static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));

      float loss = mlp.forward(d_batch_x, d_batch_t, BATCH_SIZE);
      mlp.backward(d_batch_x, d_batch_t, BATCH_SIZE);
      mlp.sgd_step(LR);
      epoch_loss += loss;
      ++n_batches;
    }

    if (epoch % 10 == 0)
      std::printf("Epoch %3d  avg batch loss: %.4f\n", epoch,
                  static_cast<double>(epoch_loss / static_cast<float>(n_batches)));
  }

  // --- Evaluate accuracy ---
  // Predict the entire dataset in one batched call (or in chunks).
  std::vector<int> all_preds;
  // Process in batches of BATCH_SIZE.
  int correct = 0;
  for (int start = 0; start < N_TOTAL; start += BATCH_SIZE) {
    int B = std::min(BATCH_SIZE, N_TOTAL - start);
    if (B < BATCH_SIZE) break;  // skip incomplete last batch for simplicity
    float* ptr_x = d_data + start * 4;
    std::vector<int> preds;
    mlp.predict_batch(ptr_x, B, preds);
    for (int b = 0; b < B; ++b)
      if (preds[static_cast<size_t>(b)] == labels[static_cast<size_t>(start + b)]) ++correct;
  }
  int evaluated = (N_TOTAL / BATCH_SIZE) * BATCH_SIZE;
  std::printf("Accuracy (mini-batch model): %d / %d (%.1f%%)\n", correct, evaluated,
              100.0 * static_cast<double>(correct) / static_cast<double>(evaluated));

  // --- Cleanup ---
  mlp.free_all();
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_targets));
  CUDA_CHECK(cudaFree(d_batch_x));
  CUDA_CHECK(cudaFree(d_batch_t));
  return EXIT_SUCCESS;
}
