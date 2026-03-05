/**
 * @file transformer.cu
 * @brief Lesson 32 — Transformer Encoder Block (Capstone).
 *
 * This capstone lesson assembles every deep-learning building block from
 * lessons 23–31 into a **single Transformer encoder block** followed by a
 * classification head, then trains it end-to-end on a synthetic task with
 * a full backward pass and Adam optimizer.
 *
 * ## Architecture
 *
 *     x = Embedding(ids) + SinusoidalPE           (Lesson 28)
 *     a = LayerNorm(x + MultiHeadSelfAttention(x)) (Lessons 29, 31)
 *     y = LayerNorm(a + FFN(a))                    (Lessons 29, 12/13)
 *     logits = cls_pool(y) @ W_cls                 (CLS pooling)
 *
 * FFN consists of two linear layers with **GELU** activation:
 *
 *     FFN(x) = GELU(x @ W1^T) @ W2^T
 *
 * GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))
 *
 * ## Training
 *
 * The backward pass reverses the forward computation:
 *   1. Cross-entropy gradient on logits
 *   2. Classification head backward (cuBLAS)
 *   3. CLS pool backward (scatter)
 *   4. LayerNorm backward
 *   5. Residual backward (gradient splits to both branches)
 *   6. FFN backward: Linear2, GELU derivative, Linear1 (cuBLAS)
 *   7. Attention backward: output projection, softmax Jacobian,
 *      QKV projections (cuBLAS + batched GEMM)
 *   8. Embedding backward (atomicAdd scatter)
 *
 * We use the Adam optimizer (Lesson 25) on all linear weight matrices.
 * LayerNorm gamma/beta are frozen (left at 1/0).
 *
 * ## Parts
 *
 * - Part 1: Forward kernels (GELU, CLS pool, softmax, embeddings, LN, etc.)
 * - Part 2: Backward kernels (GELU', softmax Jacobian, LN backward, etc.)
 * - Part 3: TransformerEncoder struct (forward + backward + Adam)
 * - Part 4: Training loop with decreasing loss
 *
 * See individual lessons for detailed explanations of each component.
 */

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
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

#define CUBLAS_CHECK(call)                                                          \
  do {                                                                              \
    cublasStatus_t st_ = (call);                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                             \
      std::fprintf(stderr, "cuBLAS error at %s:%d — code %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                          \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

constexpr int kBlockSize = 256;

/// Negative-infinity sentinel for softmax stability.
constexpr float kNegInf = -1e30F;

// =============================================================================
// Part 1 — Forward Kernels
// =============================================================================

/**
 * @brief GELU activation: x·Φ(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
 *
 * Element-wise approximate Gaussian Error Linear Unit.
 *
 * @param in  Input tensor (length @p n).
 * @param out Output tensor (length @p n).
 * @param n   Total number of elements.
 */
__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  constexpr float kSqrt2OverPi = 0.7978845608F;  // sqrt(2/pi)
  float inner = kSqrt2OverPi * (x + 0.044715F * x * x * x);
  out[i] = 0.5F * x * (1.0F + tanhf(inner));
}

/**
 * @brief CLS pooling: extract the first token (t=0) from each batch element
 *        for classification.
 *
 * @param in  Input tensor of shape (B, T, D).
 * @param out Output tensor of shape (B, D).
 * @param B   Batch size.
 * @param T   Sequence length.
 * @param D   Embedding / hidden dimension.
 */
__global__ void cls_pool_kernel(const float* __restrict__ in, float* __restrict__ out, int B, int T,
                                int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * D) return;
  int b = idx / D;
  int d = idx % D;
  out[b * D + d] = in[b * T * D + d];  // t = 0
}

/**
 * @brief Row-wise softmax with max-subtraction for numerical stability.
 *
 * Each row is handled by one block using shared-memory reductions.
 *
 * @param data Pointer to row-major matrix (rows × cols), modified in-place.
 * @param rows Number of rows.
 * @param cols Number of columns per row.
 */
__global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float* rd = data + row * cols;
  float lmax = kNegInf;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) lmax = fmaxf(lmax, rd[c]);
  smem[threadIdx.x] = lmax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float mx = smem[0];
  float lsum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf(rd[c] - mx);
    rd[c] = e;
    lsum += e;
  }
  smem[threadIdx.x] = lsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float tot = smem[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) rd[c] /= tot;
}

/**
 * @brief Cross-entropy gradient: dlogits = (softmax(x) − one_hot(label)) / B.
 *
 * Computes the per-element gradient used to kick off the backward pass.
 *
 * @param probs   Softmax probabilities, shape (B, C).
 * @param labels  Ground-truth class indices, length B.
 * @param dlogits Output gradient tensor, shape (B, C) — written in-place.
 * @param B       Batch size.
 * @param C       Number of classes.
 */
__global__ void cross_entropy_grad_kernel(const float* probs, const int* __restrict__ labels,
                                          float* dlogits, int B, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * C) return;
  int b = idx / C;
  int c = idx % C;
  float indicator = (c == labels[b]) ? 1.0F : 0.0F;
  dlogits[idx] = (probs[idx] - indicator) / static_cast<float>(B);
}

// ---- Embedding + Positional Encoding (Lesson 28) ----

/**
 * @brief Token embedding table lookup with bounds-checked assertion.
 *
 * Maps each token id to its D-dimensional embedding vector.
 *
 * @param table Embedding weight table of shape (V, D).
 * @param ids   Token indices, length @p total.
 * @param out   Output embeddings of shape (total, D).
 * @param total Number of tokens (B × T).
 * @param D     Embedding dimension.
 * @param V     Vocabulary size (used for bounds assertion).
 */
__global__ void embedding_forward_kernel(const float* __restrict__ table,
                                         const int* __restrict__ ids, float* __restrict__ out,
                                         int total, int D, int V) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  int token_id = ids[row];
  assert(token_id >= 0 && token_id < V);
  out[row * D + col] = table[token_id * D + col];
}

/**
 * @brief Add sinusoidal positional encoding to token embeddings.
 *
 * PE(pos, 2i)   = sin(pos / 10000^(2i/D))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
 *
 * @param data   Embedding tensor (total × D), modified in-place.
 * @param T      Sequence length.
 * @param D      Embedding dimension.
 * @param total  Total number of tokens (B × T).
 */
__global__ void sinusoidal_pe_kernel(float* __restrict__ data, int T, int D, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  int pos = row % T;
  float exponent = static_cast<float>(col / 2 * 2) / static_cast<float>(D);
  float freq = 1.0F / powf(10000.0F, exponent);
  float angle = static_cast<float>(pos) * freq;
  data[idx] += (col % 2 == 0) ? sinf(angle) : cosf(angle);
}

// ---- Residual + LayerNorm (Lesson 29) ----

/**
 * @brief Element-wise residual addition: y = x + residual.
 *
 * @param x         Input tensor.
 * @param residual  Residual tensor.
 * @param y         Output tensor.
 * @param n         Number of elements.
 */
__global__ void residual_add_kernel(const float* __restrict__ x, const float* __restrict__ residual,
                                    float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + residual[i];
}

/**
 * @brief Layer normalization: y = γ · (x − μ) / √(σ² + ε) + β.
 *
 * Each row is normalized independently; one block per row.
 *
 * @param x     Input tensor, shape (N, D).
 * @param y     Output tensor, shape (N, D).
 * @param gamma Scale parameter, length D.
 * @param beta  Shift parameter, length D.
 * @param mean  Per-row mean output, length N.
 * @param rstd  Per-row reciprocal std-dev output, length N.
 * @param N     Number of rows.
 * @param D     Row width (feature dimension).
 * @param eps   Small constant for numerical stability.
 */
__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ mean,
                                         float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;
  const float* ri = x + row * D;
  extern __shared__ float smem[];
  float ls = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) ls += ri[d];
  smem[threadIdx.x] = ls;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);
  float lv = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float diff = ri[d] - mu;
    lv += diff * diff;
  }
  smem[threadIdx.x] = lv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }
  float* ro = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float xh = (ri[d] - mu) * inv_std;
    ro[d] = gamma[d] * xh + beta[d];
  }
}

// ---- Self-Attention helpers (Lesson 31) ----

/**
 * @brief Reshape tensor from (B, T, D) layout to (B, nH, T, dK) for
 *        multi-head attention.
 *
 * @param in        Input data with row stride @p in_stride.
 * @param out       Output in per-head layout (B, nH, T, dK).
 * @param B         Batch size.
 * @param T         Sequence length.
 * @param nH        Number of attention heads.
 * @param dK        Head dimension (D / nH).
 * @param in_stride Row stride of input (D for contiguous, 3D for QKV buf).
 */
__global__ void split_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK, int in_stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int in_idx = b * (T * in_stride) + t * in_stride + h * dK + dk;
  int out_idx = ((b * nH + h) * T + t) * dK + dk;
  out[out_idx] = in[in_idx];
}

/**
 * @brief Inverse of split_heads: reshape (B, nH, T, dK) → (B, T, D).
 *
 * Merges per-head outputs back into a contiguous hidden-dimension layout.
 *
 * @param in         Input in per-head layout (B, nH, T, dK).
 * @param out        Output data with row stride @p out_stride.
 * @param B          Batch size.
 * @param T          Sequence length.
 * @param nH         Number of attention heads.
 * @param dK         Head dimension (D / nH).
 * @param out_stride Row stride of output (D for contiguous, 3D for QKV buf).
 */
__global__ void merge_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK, int out_stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;
  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);
  int in_idx = ((b * nH + h) * T + t) * dK + dk;
  int out_idx = b * (T * out_stride) + t * out_stride + h * dK + dk;
  out[out_idx] = in[in_idx];
}

// =============================================================================
// Part 2 — Backward Kernels
// =============================================================================

/**
 * @brief GELU backward: grad_in = grad_out * gelu'(x).
 *
 * gelu'(x) = 0.5*(1+tanh(z)) + 0.5*x*sech^2(z)*(sqrt(2/pi)*(1+3*0.044715*x^2))
 * where z = sqrt(2/pi)*(x + 0.044715*x^3).
 *
 * @param grad_out Incoming gradient tensor (length @p n).
 * @param x_in     Original forward-pass input (length @p n).
 * @param grad_in  Output gradient tensor (length @p n).
 * @param n        Total number of elements.
 */
__global__ void gelu_backward_kernel(const float* grad_out, const float* __restrict__ x_in,
                                     float* grad_in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = x_in[i];
  constexpr float kA = 0.7978845608F;  // sqrt(2/pi)
  constexpr float kB = 0.044715F;
  float inner = kA * (x + kB * x * x * x);
  float th = tanhf(inner);
  float sech2 = 1.0F - th * th;
  float d_inner = kA * (1.0F + 3.0F * kB * x * x);
  float gelu_prime = 0.5F * (1.0F + th) + 0.5F * x * sech2 * d_inner;
  grad_in[i] = grad_out[i] * gelu_prime;
}

/**
 * @brief CLS pool backward: scatter gradient from (B,D) to position 0 in
 *        each (T,D) block.  Output buffer must be pre-zeroed.
 *
 * @param grad_out Gradient from classification head, shape (B, D).
 * @param grad_in  Output gradient tensor, shape (B, T, D) — must be pre-zeroed.
 * @param B        Batch size.
 * @param T        Sequence length.
 * @param D        Hidden dimension.
 */
__global__ void cls_pool_backward_kernel(const float* __restrict__ grad_out,
                                         float* __restrict__ grad_in, int B, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * D) return;
  int b = idx / D;
  int d = idx % D;
  grad_in[b * T * D + d] = grad_out[b * D + d];  // position t=0
}

/**
 * @brief Softmax backward (Jacobian-vector product), one row per block.
 *
 * For row i:  d_in_j = S_j * (d_out_j - sum_k(d_out_k * S_k))
 *
 * @param grad_out    Incoming gradient, shape (rows, cols).
 * @param softmax_out Cached softmax output from forward pass, shape (rows, cols).
 * @param grad_in     Output gradient, shape (rows, cols).
 * @param rows        Number of rows.
 * @param cols        Number of columns per row.
 */
__global__ void softmax_backward_kernel(const float* grad_out,
                                        const float* __restrict__ softmax_out, float* grad_in,
                                        int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  const float* dO = grad_out + row * cols;
  const float* S = softmax_out + row * cols;
  float* dI = grad_in + row * cols;

  // dot = sum(dO * S)
  float local_dot = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_dot += dO[c] * S[c];
  smem[threadIdx.x] = local_dot;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float dot = smem[0];

  // d_in_j = S_j * (dO_j - dot)
  for (int c = threadIdx.x; c < cols; c += blockDim.x) dI[c] = S[c] * (dO[c] - dot);
}

/**
 * @brief LayerNorm backward: compute dx given dy, x, gamma, mean, rstd.
 *
 * dx_i = rstd * (gamma_i * dy_i - c1 - x_hat_i * c2)
 * where c1 = mean(gamma * dy), c2 = mean(gamma * dy * x_hat).
 * Gamma/beta gradients are not computed (frozen).
 *
 * @param dy    Incoming gradient, shape (N, D).
 * @param x     Original input from forward pass, shape (N, D).
 * @param gamma Scale parameter, length D.
 * @param mean  Per-row mean cached from forward pass, length N.
 * @param rstd  Per-row reciprocal std-dev from forward pass, length N.
 * @param dx    Output gradient, shape (N, D).
 * @param N     Number of rows.
 * @param D     Row width (feature dimension).
 */
__global__ void layernorm_backward_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ rstd, float* __restrict__ dx,
                                          int N, int D) {
  int row = blockIdx.x;
  if (row >= N) return;
  extern __shared__ float smem[];
  const float* dy_row = dy + row * D;
  const float* x_row = x + row * D;
  float mu = mean[row];
  float rs = rstd[row];
  float inv_D = 1.0F / static_cast<float>(D);

  // Accumulate c1 = sum(gamma * dy) / D, c2 = sum(gamma * dy * x_hat) / D
  float s1 = 0.0F, s2 = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float xh = (x_row[d] - mu) * rs;
    float gd = gamma[d] * dy_row[d];
    s1 += gd;
    s2 += gd * xh;
  }

  // Reduce s1
  smem[threadIdx.x] = s1;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float c1 = smem[0] * inv_D;
  __syncthreads();

  // Reduce s2
  smem[threadIdx.x] = s2;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float c2 = smem[0] * inv_D;

  // Write dx
  float* dx_row = dx + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float xh = (x_row[d] - mu) * rs;
    dx_row[d] = rs * (gamma[d] * dy_row[d] - c1 - xh * c2);
  }
}

/**
 * @brief Embedding backward: scatter gradients to the embedding table
 *        using atomicAdd (multiple tokens may map to the same row).
 *
 * @param grad    Incoming gradient, shape (total, D).
 * @param ids     Token indices, length @p total.
 * @param d_table Gradient accumulator for embedding table, shape (V, D).
 * @param total   Number of tokens (B × T).
 * @param D       Embedding dimension.
 */
__global__ void embedding_backward_kernel(const float* __restrict__ grad,
                                          const int* __restrict__ ids, float* __restrict__ d_table,
                                          int total, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;
  int row = idx / D;
  int col = idx % D;
  atomicAdd(&d_table[ids[row] * D + col], grad[row * D + col]);
}

// =============================================================================
// Adam Optimizer (Lesson 25)
// =============================================================================

/**
 * @brief Adam optimizer update step with bias-corrected first and second
 *        moment estimates.
 *
 * @param params Parameter tensor to update in-place (length @p n).
 * @param grads  Gradient tensor (length @p n).
 * @param m      First-moment estimate (length @p n), updated in-place.
 * @param v      Second-moment estimate (length @p n), updated in-place.
 * @param n      Total number of parameters.
 * @param lr     Learning rate.
 * @param beta1  Exponential decay rate for first moment.
 * @param beta2  Exponential decay rate for second moment.
 * @param eps    Small constant for numerical stability.
 * @param t      Current time step (1-based) for bias correction.
 */
__global__ void adam_step_kernel(float* __restrict__ params, const float* __restrict__ grads,
                                 float* __restrict__ m, float* __restrict__ v, int n, float lr,
                                 float beta1, float beta2, float eps, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = grads[i];
  m[i] = beta1 * m[i] + (1.0F - beta1) * g;
  v[i] = beta2 * v[i] + (1.0F - beta2) * g * g;
  float m_hat = m[i] / (1.0F - powf(beta1, static_cast<float>(t)));
  float v_hat = v[i] / (1.0F - powf(beta2, static_cast<float>(t)));
  params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// =============================================================================
// Part 3 — TransformerEncoder
// =============================================================================

/**
 * @brief Single-layer Transformer encoder with embedding, multi-head
 *        self-attention, feed-forward network (FFN), and a linear
 *        classification head.  Supports end-to-end forward / backward
 *        and Adam parameter updates.
 */
struct TransformerEncoder {
  // Hyper-params
  int V, D, T, nH, dK, dFF, nC;
  cublasHandle_t handle;
  int B_max;

  // ---- Weights (device) ----
  float* emb_table;  // (V x D)
  float* W_QKV;      // (3D x D)
  float* W_O;        // (D x D)
  float* ln1_gamma;  // (D)
  float* ln1_beta;   // (D)
  float* W1;         // (dFF x D)
  float* b1;         // (dFF)  — unused (zero), kept for symmetry
  float* W2;         // (D x dFF)
  float* b2;         // (D)    — unused (zero), kept for symmetry
  float* ln2_gamma;  // (D)
  float* ln2_beta;   // (D)
  float* W_cls;      // (nC x D)
  float* b_cls;      // (nC)   — unused (zero), kept for symmetry

  // ---- Forward scratch (device) ----
  float* x_emb;    // (B x T x D)
  float* qkv_buf;  // (B x T x 3D)
  float* Q;        // (B x nH x T x dK)
  float* K;
  float* Vb;
  float* scores;       // (B x nH x T x T) — post-softmax = attn weights
  float* ctx;          // (B x nH x T x dK)
  float* attn_merged;  // (B x T x D) — post merge_heads, pre W_O
  float* attn_out;     // (B x T x D) — post W_O
  float* res1;         // (B x T x D)
  float* ln1_out;      // (B x T x D)
  float* ln1_mean;     // (B x T)
  float* ln1_rstd;     // (B x T)
  float* ffn_mid;      // (B x T x dFF) — pre-GELU
  float* ffn_gelu;     // (B x T x dFF) — post-GELU
  float* ffn_out;      // (B x T x D)
  float* res2;         // (B x T x D)
  float* ln2_out;      // (B x T x D)
  float* ln2_mean;
  float* ln2_rstd;
  float* cls_in;  // (B x D)
  float* logits;  // (B x nC)

  // ---- Backward scratch (device) ----
  float* grad1;        // (BT x D)   — reusable scratch
  float* grad2;        // (BT x D)   — reusable scratch / accumulator
  float* grad_dff;     // (BT x dFF) — FFN mid gradients
  float* grad_scores;  // (B x nH x T x T) — attention score gradients
  float* grad_bd;      // (B x D)    — classification head gradients
  float* grad_qkv;     // (BT x 3D)  — QKV projection gradients

  // ---- Parameter gradients & optimizer state ----
  static constexpr int kNP = 6;  // emb, QKV, O, W1, W2, cls
  float* pgrad[kNP];             // parameter gradients
  float* adam_m[kNP];            // Adam first moment
  float* adam_v[kNP];            // Adam second moment

  // =========================================================================
  // Forward
  // =========================================================================

  /**
   * @brief Execute the full forward pass: embedding → attention → FFN →
   *        classification logits.
   *
   * @param d_ids Device pointer to token indices, shape (B, T).
   * @param B     Batch size (must be ≤ B_max).
   */
  void forward(const int* d_ids, int B) {
    float alpha = 1.0F, beta_z = 0.0F;
    constexpr float kEps = 1e-5F;
    int BT = B * T;
    int BTD = BT * D;

    // ---- Embedding + PE ----
    int grid_emb = (BT * D + kBlockSize - 1) / kBlockSize;
    embedding_forward_kernel<<<grid_emb, kBlockSize>>>(emb_table, d_ids, x_emb, BT, D, V);
    CUDA_CHECK(cudaGetLastError());
    sinusoidal_pe_kernel<<<grid_emb, kBlockSize>>>(x_emb, T, D, BT);
    CUDA_CHECK(cudaGetLastError());

    // ---- Self-Attention ----
    // QKV projection: qkv_buf(BT,3D) = x_emb(BT,D) @ W_QKV(3D,D)^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * D, BT, D, &alpha, W_QKV, D,
                             x_emb, D, &beta_z, qkv_buf, 3 * D));

    // Split interleaved QKV into per-head layout  (stride = 3D in qkv_buf)
    int grid_h = (BTD + kBlockSize - 1) / kBlockSize;
    split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf, Q, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf + D, K, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    split_heads_kernel<<<grid_h, kBlockSize>>>(qkv_buf + 2 * D, Vb, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());

    // Attention scores: scores(T,T) = Q(T,dK) @ K(T,dK)^T * scale  (per head)
    float scale = 1.0F / sqrtf(static_cast<float>(dK));
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dK, &scale, K,
                                           dK, T * dK, Q, dK, T * dK, &beta_z, scores, T, T * T,
                                           B * nH));

    // Row-wise softmax on scores → attention weights  (in-place)
    int smem_blk = 128;
    softmax_kernel<<<B * nH * T, smem_blk, smem_blk * sizeof(float)>>>(scores, B * nH * T, T);
    CUDA_CHECK(cudaGetLastError());

    // Context: ctx(T,dK) = attn(T,T) @ V(T,dK)  (per head)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dK, T, T, &alpha, Vb,
                                           dK, T * dK, scores, T, T * T, &beta_z, ctx, dK, T * dK,
                                           B * nH));

    // Merge heads → output projection
    merge_heads_kernel<<<grid_h, kBlockSize>>>(ctx, attn_merged, B, T, nH, dK, D);
    CUDA_CHECK(cudaGetLastError());

    // attn_out(BT,D) = attn_merged(BT,D) @ W_O(D,D)^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, BT, D, &alpha, W_O, D,
                             attn_merged, D, &beta_z, attn_out, D));

    // ---- Residual + LN 1 ----
    int grid_add = (BTD + kBlockSize - 1) / kBlockSize;
    residual_add_kernel<<<grid_add, kBlockSize>>>(attn_out, x_emb, res1, BTD);
    CUDA_CHECK(cudaGetLastError());

    int ln_blk = 128;
    layernorm_forward_kernel<<<BT, ln_blk, ln_blk * sizeof(float)>>>(
        res1, ln1_out, ln1_gamma, ln1_beta, ln1_mean, ln1_rstd, BT, D, kEps);
    CUDA_CHECK(cudaGetLastError());

    // ---- FFN: GELU(x @ W1^T) @ W2^T ----
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dFF, BT, D, &alpha, W1, D, ln1_out,
                             D, &beta_z, ffn_mid, dFF));

    int gelu_grid = (BT * dFF + kBlockSize - 1) / kBlockSize;
    gelu_kernel<<<gelu_grid, kBlockSize>>>(ffn_mid, ffn_gelu, BT * dFF);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, BT, dFF, &alpha, W2, dFF,
                             ffn_gelu, dFF, &beta_z, ffn_out, D));

    // ---- Residual + LN 2 ----
    residual_add_kernel<<<grid_add, kBlockSize>>>(ffn_out, ln1_out, res2, BTD);
    CUDA_CHECK(cudaGetLastError());
    layernorm_forward_kernel<<<BT, ln_blk, ln_blk * sizeof(float)>>>(
        res2, ln2_out, ln2_gamma, ln2_beta, ln2_mean, ln2_rstd, BT, D, kEps);
    CUDA_CHECK(cudaGetLastError());

    // ---- CLS pooling + classification ----
    int cls_grid = (B * D + kBlockSize - 1) / kBlockSize;
    cls_pool_kernel<<<cls_grid, kBlockSize>>>(ln2_out, cls_in, B, T, D);
    CUDA_CHECK(cudaGetLastError());

    // logits(B,nC) = cls_in(B,D) @ W_cls(nC,D)^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nC, B, D, &alpha, W_cls, D, cls_in,
                             D, &beta_z, logits, nC));
  }

  // =========================================================================
  // Backward
  // =========================================================================

  /**
   * @brief Full backward pass.  Assumes forward() was called first and that
   *        @p d_dlogits already contains softmax(logits) (probabilities).
   *        The method overwrites d_dlogits with the cross-entropy gradient,
   *        then backpropagates through every layer, accumulating parameter
   *        gradients in pgrad[].
   *
   * @param d_ids     Device pointer to token indices, shape (B, T).
   * @param d_labels  Device pointer to ground-truth class labels, length B.
   * @param d_dlogits Logit probabilities buffer, shape (B, nC) — overwritten
   *                  with cross-entropy gradient, then consumed.
   * @param B         Batch size.
   */
  void backward(const int* d_ids, const int* d_labels, float* d_dlogits, int B) {
    float alpha = 1.0F, beta_z = 0.0F, beta_one = 1.0F;
    int BT = B * T;
    int BTD = BT * D;
    float scale = 1.0F / sqrtf(static_cast<float>(dK));
    int ln_blk = 128;

    // ---- 1. Cross-entropy gradient: dlogits = (probs - one_hot) / B ----
    int ce_grid = (B * nC + kBlockSize - 1) / kBlockSize;
    cross_entropy_grad_kernel<<<ce_grid, kBlockSize>>>(d_dlogits, d_labels, d_dlogits, B, nC);
    CUDA_CHECK(cudaGetLastError());

    // ---- 2. Classification head backward ----
    // dW_cls(nC,D) = dlogits(B,nC)^T @ cls_in(B,D)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, nC, B, &alpha, cls_in, D,
                             d_dlogits, nC, &beta_z, pgrad[5], D));
    // d_cls_in(B,D) = dlogits(B,nC) @ W_cls(nC,D)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, B, nC, &alpha, W_cls, D,
                             d_dlogits, nC, &beta_z, grad_bd, D));

    // ---- 3. CLS pool backward ----
    CUDA_CHECK(cudaMemset(grad1, 0, BTD * sizeof(float)));
    int cls_grid = (B * D + kBlockSize - 1) / kBlockSize;
    cls_pool_backward_kernel<<<cls_grid, kBlockSize>>>(grad_bd, grad1, B, T,
                                                       D);  // grad1 = d_ln2_out
    CUDA_CHECK(cudaGetLastError());

    // ---- 4. LayerNorm 2 backward ----
    layernorm_backward_kernel<<<BT, ln_blk, ln_blk * sizeof(float)>>>(
        grad1, res2, ln2_gamma, ln2_mean, ln2_rstd, grad2, BT, D);
    CUDA_CHECK(cudaGetLastError());
    // grad2 = d_res2

    // ---- 5. Residual 2 backward: d_ffn_out = d_res2, d_ln1 = d_res2 ----
    CUDA_CHECK(cudaMemcpy(grad1, grad2, BTD * sizeof(float), cudaMemcpyDeviceToDevice));
    // grad1 = d_ffn_out
    // grad2 = d_ln1_out (accumulator)

    // ---- 6. Linear2 backward ----
    // dW2(D,dFF) = d_ffn_out^T @ ffn_gelu
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dFF, D, BT, &alpha, ffn_gelu, dFF,
                             grad1, D, &beta_z, pgrad[4], dFF));
    // d_ffn_gelu(BT,dFF) = d_ffn_out @ W2
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dFF, BT, D, &alpha, W2, dFF, grad1,
                             D, &beta_z, grad_dff, dFF));

    // ---- 7. GELU backward ----
    int gelu_grid = (BT * dFF + kBlockSize - 1) / kBlockSize;
    gelu_backward_kernel<<<gelu_grid, kBlockSize>>>(grad_dff, ffn_mid, grad_dff,
                                                    BT * dFF);  // grad_dff = d_ffn_mid
    CUDA_CHECK(cudaGetLastError());

    // ---- 8. Linear1 backward ----
    // dW1(dFF,D) = d_ffn_mid^T @ ln1_out
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, dFF, BT, &alpha, ln1_out, D,
                             grad_dff, dFF, &beta_z, pgrad[3], D));
    // d_ln1_out += d_ffn_mid @ W1  (accumulate with residual branch)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, BT, dFF, &alpha, W1, D, grad_dff,
                             dFF, &beta_one, grad2, D));
    // grad2 = total d_ln1_out

    // ---- 9. LayerNorm 1 backward ----
    layernorm_backward_kernel<<<BT, ln_blk, ln_blk * sizeof(float)>>>(
        grad2, res1, ln1_gamma, ln1_mean, ln1_rstd, grad1, BT, D);
    CUDA_CHECK(cudaGetLastError());
    // grad1 = d_res1

    // ---- 10. Residual 1 backward: d_attn_out = d_res1, d_x_emb = d_res1 ----
    CUDA_CHECK(cudaMemcpy(grad2, grad1, BTD * sizeof(float), cudaMemcpyDeviceToDevice));
    // grad1 = d_attn_out
    // grad2 = d_x_emb (accumulator)

    // ---- 11. Output projection backward ----
    // dW_O(D,D) = d_attn_out^T @ attn_merged
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, BT, &alpha, attn_merged, D,
                             grad1, D, &beta_z, pgrad[2], D));
    // d_merged(BT,D) = d_attn_out @ W_O
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, BT, D, &alpha, W_O, D, grad1, D,
                             &beta_z, grad1, D));
    // grad1 = d_merged

    // ---- 12. Merge-heads backward = split heads on the gradient ----
    int grid_h = (BTD + kBlockSize - 1) / kBlockSize;
    split_heads_kernel<<<grid_h, kBlockSize>>>(grad1, ctx, B, T, nH, dK, D);
    CUDA_CHECK(cudaGetLastError());
    // ctx = d_ctx (head layout)

    // ---- 13. Context backward: ctx = attn @ V ----
    // d_attn(T,T) = V^T(dK,T)^T @ d_ctx  → per head
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dK, &alpha, Vb,
                                           dK, T * dK, ctx, dK, T * dK, &beta_z, grad_scores, T,
                                           T * T, B * nH));  // grad_scores = d_attn

    // d_V(T,dK) = attn^T @ d_ctx  → per head  (reuse Vb for d_V)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, dK, T, T, &alpha, ctx,
                                           dK, T * dK, scores, T, T * T, &beta_z, Vb, dK, T * dK,
                                           B * nH));  // Vb = d_V

    // ---- 14. Softmax backward ----
    int smem_blk = 128;
    softmax_backward_kernel<<<B * nH * T, smem_blk, smem_blk * sizeof(float)>>>(
        grad_scores, scores, grad_scores, B * nH * T, T);
    CUDA_CHECK(cudaGetLastError());
    // grad_scores = d_raw_scores

    // ---- 15. Score backward: scores = Q @ K^T * scale ----
    // d_Q(T,dK) = d_scores(T,T) @ K(T,dK) * scale  (→ store in ctx)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dK, T, T, &scale, K,
                                           dK, T * dK, grad_scores, T, T * T, &beta_z, ctx, dK,
                                           T * dK, B * nH));  // ctx = d_Q

    // d_K(T,dK) = d_scores^T(T,T) @ Q(T,dK) * scale  (→ store in K)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, dK, T, T, &scale, Q,
                                           dK, T * dK, grad_scores, T, T * T, &beta_z, K, dK,
                                           T * dK, B * nH));  // K = d_K

    // ---- 16. Split-heads backward = merge heads ----
    // Scatter d_Q, d_K, d_V back to interleaved grad_qkv (stride 3D)
    CUDA_CHECK(cudaMemset(grad_qkv, 0, BT * 3 * D * sizeof(float)));
    merge_heads_kernel<<<grid_h, kBlockSize>>>(ctx, grad_qkv, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    merge_heads_kernel<<<grid_h, kBlockSize>>>(K, grad_qkv + D, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    merge_heads_kernel<<<grid_h, kBlockSize>>>(Vb, grad_qkv + 2 * D, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());

    // ---- 17. QKV projection backward ----
    // dW_QKV(3D,D) = d_qkv^T @ x_emb
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, D, 3 * D, BT, &alpha, x_emb, D,
                             grad_qkv, 3 * D, &beta_z, pgrad[1], D));
    // d_x_emb += d_qkv @ W_QKV  (accumulate with residual branch)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, BT, 3 * D, &alpha, W_QKV, D,
                             grad_qkv, 3 * D, &beta_one, grad2, D));  // grad2 = total d_x_emb

    // ---- 18. Embedding backward ----
    CUDA_CHECK(cudaMemset(pgrad[0], 0, V * D * sizeof(float)));
    int emb_grid = (BT * D + kBlockSize - 1) / kBlockSize;
    embedding_backward_kernel<<<emb_grid, kBlockSize>>>(grad2, d_ids, pgrad[0], BT, D);
    CUDA_CHECK(cudaGetLastError());
  }

  // =========================================================================
  // Optimiser helpers
  // =========================================================================

  /** @brief Zero all parameter-gradient buffers before the next backward pass. */
  void zero_grads() {
    int sizes[kNP] = {V * D, 3 * D * D, D * D, dFF * D, D * dFF, nC * D};
    for (int i = 0; i < kNP; ++i) CUDA_CHECK(cudaMemset(pgrad[i], 0, sizes[i] * sizeof(float)));
  }

  void adam_update(float lr, int step) {
    float* params[kNP] = {emb_table, W_QKV, W_O, W1, W2, W_cls};
    int sizes[kNP] = {V * D, 3 * D * D, D * D, dFF * D, D * dFF, nC * D};
    for (int i = 0; i < kNP; ++i) {
      int grid = (sizes[i] + kBlockSize - 1) / kBlockSize;
      adam_step_kernel<<<grid, kBlockSize>>>(params[i], pgrad[i], adam_m[i], adam_v[i], sizes[i],
                                             lr, 0.9F, 0.999F, 1e-8F, step);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

// =============================================================================
// Part 4 — Training Loop
// =============================================================================

int main() {
  // Hyper-params
  constexpr int kV = 32;
  constexpr int kD = 32;
  constexpr int kT = 8;
  constexpr int kNH = 4;
  constexpr int kDK = kD / kNH;
  constexpr int kDFF = 64;
  constexpr int kNC = 2;
  constexpr int kB = 16;
  constexpr int kEpochs = 50;
  constexpr float kLR = 1e-3F;

  std::mt19937 rng(42);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---- Helper lambdas for allocation ----
  auto alloc_init = [&](float** ptr, int size, float sc) {
    CUDA_CHECK(cudaMalloc(ptr, size * sizeof(float)));
    std::vector<float> h(size);
    std::uniform_real_distribution<float> dist(-sc, sc);
    for (auto& v : h) v = dist(rng);
    CUDA_CHECK(cudaMemcpy(*ptr, h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  };

  auto alloc_const = [&](float** ptr, int size, float val) {
    CUDA_CHECK(cudaMalloc(ptr, size * sizeof(float)));
    std::vector<float> h(size, val);
    CUDA_CHECK(cudaMemcpy(*ptr, h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  };

  auto alloc_zero = [&](float** ptr, int size) {
    CUDA_CHECK(cudaMalloc(ptr, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(*ptr, 0, size * sizeof(float)));
  };

  float fan = 1.0F / sqrtf(static_cast<float>(kD));

  // ---- Build encoder ----
  TransformerEncoder enc{};
  enc.V = kV;
  enc.D = kD;
  enc.T = kT;
  enc.nH = kNH;
  enc.dK = kDK;
  enc.dFF = kDFF;
  enc.nC = kNC;
  enc.handle = handle;
  enc.B_max = kB;

  // Weights
  alloc_init(&enc.emb_table, kV * kD, fan);
  alloc_init(&enc.W_QKV, 3 * kD * kD, fan);
  alloc_init(&enc.W_O, kD * kD, fan);
  alloc_const(&enc.ln1_gamma, kD, 1.0F);
  alloc_const(&enc.ln1_beta, kD, 0.0F);
  alloc_init(&enc.W1, kDFF * kD, fan);
  alloc_const(&enc.b1, kDFF, 0.0F);
  alloc_init(&enc.W2, kD * kDFF, 1.0F / sqrtf(static_cast<float>(kDFF)));
  alloc_const(&enc.b2, kD, 0.0F);
  alloc_const(&enc.ln2_gamma, kD, 1.0F);
  alloc_const(&enc.ln2_beta, kD, 0.0F);
  alloc_init(&enc.W_cls, kNC * kD, fan);
  alloc_const(&enc.b_cls, kNC, 0.0F);

  // Forward scratch
  int BT = kB * kT;
  CUDA_CHECK(cudaMalloc(&enc.x_emb, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.qkv_buf, BT * 3 * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.Q, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.K, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.Vb, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.scores, kB * kNH * kT * kT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ctx, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.attn_merged, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.attn_out, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.res1, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln1_out, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln1_mean, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln1_rstd, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ffn_mid, BT * kDFF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ffn_gelu, BT * kDFF * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ffn_out, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.res2, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln2_out, BT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln2_mean, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.ln2_rstd, BT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.cls_in, kB * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&enc.logits, kB * kNC * sizeof(float)));

  // Backward scratch
  alloc_zero(&enc.grad1, BT * kD);
  alloc_zero(&enc.grad2, BT * kD);
  alloc_zero(&enc.grad_dff, BT * kDFF);
  alloc_zero(&enc.grad_scores, kB * kNH * kT * kT);
  alloc_zero(&enc.grad_bd, kB * kD);
  alloc_zero(&enc.grad_qkv, BT * 3 * kD);

  // Parameter gradients + Adam state
  constexpr int kSizes[TransformerEncoder::kNP] = {kV * kD,   3 * kD * kD, kD * kD,
                                                   kDFF * kD, kD * kDFF,   kNC * kD};
  for (int i = 0; i < TransformerEncoder::kNP; ++i) {
    alloc_zero(&enc.pgrad[i], kSizes[i]);
    alloc_zero(&enc.adam_m[i], kSizes[i]);
    alloc_zero(&enc.adam_v[i], kSizes[i]);
  }

  // ---- Synthetic data ----
  // Task: binary classification — "does the sequence contain token 1?"
  // We generate one fixed dataset and train on it to show convergence.
  std::vector<int> all_ids(kB * kT);
  std::vector<int> all_labels(kB);

  {
    std::uniform_int_distribution<int> tdist(2, kV - 1);
    for (int b = 0; b < kB; ++b) {
      bool has_one = (rng() % 2 == 0);
      all_labels[b] = has_one ? 1 : 0;
      for (int t = 0; t < kT; ++t) all_ids[b * kT + t] = tdist(rng);
      if (has_one) {
        int pos = static_cast<int>(rng() % kT);
        all_ids[b * kT + pos] = 1;
      }
    }
  }

  int* d_ids;
  int* d_labels;
  CUDA_CHECK(cudaMalloc(&d_ids, kB * kT * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_labels, kB * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, all_ids.data(), kB * kT * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, all_labels.data(), kB * sizeof(int), cudaMemcpyHostToDevice));

  // dlogits buffer (reused each epoch for probs → gradient)
  float* d_dlogits;
  CUDA_CHECK(cudaMalloc(&d_dlogits, kB * kNC * sizeof(float)));

  // ---- Training loop ----
  std::printf("=== Transformer Encoder Training ===\n");
  std::printf("  V=%d D=%d T=%d nH=%d dFF=%d nC=%d B=%d lr=%.0e\n", kV, kD, kT, kNH, kDFF, kNC, kB,
              static_cast<double>(kLR));

  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    // Forward
    enc.forward(d_ids, kB);

    // Softmax on logits → probabilities (copy first to preserve logits)
    CUDA_CHECK(
        cudaMemcpy(d_dlogits, enc.logits, kB * kNC * sizeof(float), cudaMemcpyDeviceToDevice));
    int smem_blk = 32;
    softmax_kernel<<<kB, smem_blk, smem_blk * sizeof(float)>>>(d_dlogits, kB, kNC);
    CUDA_CHECK(cudaGetLastError());

    // Compute loss & accuracy on CPU for display
    std::vector<float> h_probs(kB * kNC);
    CUDA_CHECK(
        cudaMemcpy(h_probs.data(), d_dlogits, kB * kNC * sizeof(float), cudaMemcpyDeviceToHost));
    float loss = 0.0F;
    int correct = 0;
    for (int b = 0; b < kB; ++b) {
      int lbl = all_labels[b];
      float p = h_probs[b * kNC + lbl];
      loss -= logf(fmaxf(p, 1e-7F));
      int pred = (h_probs[b * kNC + 1] > h_probs[b * kNC]) ? 1 : 0;
      if (pred == lbl) ++correct;
    }
    loss /= static_cast<float>(kB);

    if (epoch % 10 == 0 || epoch == kEpochs - 1) {
      std::printf("  Epoch %3d  loss=%.4f  acc=%.1f%%\n", epoch, static_cast<double>(loss),
                  static_cast<double>(correct) / kB * 100.0);
    }

    // Backward (d_dlogits still holds probs; backward applies CE gradient)
    enc.zero_grads();
    enc.backward(d_ids, d_labels, d_dlogits, kB);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Adam update (step is 1-based)
    enc.adam_update(kLR, epoch + 1);
  }

  // ---- Cleanup ----
  // Weights
  CUDA_CHECK(cudaFree(enc.emb_table));
  CUDA_CHECK(cudaFree(enc.W_QKV));
  CUDA_CHECK(cudaFree(enc.W_O));
  CUDA_CHECK(cudaFree(enc.ln1_gamma));
  CUDA_CHECK(cudaFree(enc.ln1_beta));
  CUDA_CHECK(cudaFree(enc.W1));
  CUDA_CHECK(cudaFree(enc.b1));
  CUDA_CHECK(cudaFree(enc.W2));
  CUDA_CHECK(cudaFree(enc.b2));
  CUDA_CHECK(cudaFree(enc.ln2_gamma));
  CUDA_CHECK(cudaFree(enc.ln2_beta));
  CUDA_CHECK(cudaFree(enc.W_cls));
  CUDA_CHECK(cudaFree(enc.b_cls));

  // Forward scratch
  CUDA_CHECK(cudaFree(enc.x_emb));
  CUDA_CHECK(cudaFree(enc.qkv_buf));
  CUDA_CHECK(cudaFree(enc.Q));
  CUDA_CHECK(cudaFree(enc.K));
  CUDA_CHECK(cudaFree(enc.Vb));
  CUDA_CHECK(cudaFree(enc.scores));
  CUDA_CHECK(cudaFree(enc.ctx));
  CUDA_CHECK(cudaFree(enc.attn_merged));
  CUDA_CHECK(cudaFree(enc.attn_out));
  CUDA_CHECK(cudaFree(enc.res1));
  CUDA_CHECK(cudaFree(enc.ln1_out));
  CUDA_CHECK(cudaFree(enc.ln1_mean));
  CUDA_CHECK(cudaFree(enc.ln1_rstd));
  CUDA_CHECK(cudaFree(enc.ffn_mid));
  CUDA_CHECK(cudaFree(enc.ffn_gelu));
  CUDA_CHECK(cudaFree(enc.ffn_out));
  CUDA_CHECK(cudaFree(enc.res2));
  CUDA_CHECK(cudaFree(enc.ln2_out));
  CUDA_CHECK(cudaFree(enc.ln2_mean));
  CUDA_CHECK(cudaFree(enc.ln2_rstd));
  CUDA_CHECK(cudaFree(enc.cls_in));
  CUDA_CHECK(cudaFree(enc.logits));

  // Backward scratch
  CUDA_CHECK(cudaFree(enc.grad1));
  CUDA_CHECK(cudaFree(enc.grad2));
  CUDA_CHECK(cudaFree(enc.grad_dff));
  CUDA_CHECK(cudaFree(enc.grad_scores));
  CUDA_CHECK(cudaFree(enc.grad_bd));
  CUDA_CHECK(cudaFree(enc.grad_qkv));

  // Parameter gradients + Adam state
  for (int i = 0; i < TransformerEncoder::kNP; ++i) {
    CUDA_CHECK(cudaFree(enc.pgrad[i]));
    CUDA_CHECK(cudaFree(enc.adam_m[i]));
    CUDA_CHECK(cudaFree(enc.adam_v[i]));
  }

  CUDA_CHECK(cudaFree(d_ids));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_dlogits));
  CUBLAS_CHECK(cublasDestroy(handle));

  std::printf("\nDone.\n");
  return EXIT_SUCCESS;
}
