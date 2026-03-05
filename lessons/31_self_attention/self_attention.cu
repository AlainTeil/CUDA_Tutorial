/**
 * @file self_attention.cu
 * @brief Lesson 31 — Multi‑Head Self‑Attention.
 *
 * Self‑attention is the core mechanism of the Transformer architecture
 * (Vaswani et al., 2017).  Given a sequence of token embeddings it
 * computes:
 *
 *     Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
 *
 * ## Multi‑Head Attention
 *
 *     MultiHead(X) = Concat(head_1, ..., head_h) W_O
 *     where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
 *
 * We split the D‑dimensional model into h heads, each of dimension
 * d_k = D / h.
 *
 * ## Parts
 *
 * - Part 1: QKV Projection — single batched GEMM via cuBLAS to
 *   compute Q, K, V simultaneously (X @ W_{QKV}^T)
 * - Part 2: Split & Reshape — rearrange (B, T, 3, h, d_k) →
 *   3 × (B, h, T, d_k)
 * - Part 3: Scaled Dot-Product Attention — score = Q @ K^T / √d_k
 *   (cuBLAS batched GEMM), softmax, context = Attn @ V
 * - Part 4: Merge Heads & Output Projection — reshape
 *   (B, h, T, d_k) → (B, T, D) and project through W_O
 *
 * See Lesson 28 for embeddings, Lesson 29 for residual + layer norm, and
 * Lesson 32 for the full Transformer encoder block that wraps this.
 */

#include <cublas_v2.h>

#include <cassert>
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
// Part 2 — Split heads: (B, T, nH*dK) → (B, nH, T, dK)
// =============================================================================

/**
 * @brief Reshape + transpose for multi‑head layout.
 *
 * Input:  in[ b ][ t ][ d ]  where d ∈ [0, nH*dK), with row stride @p in_stride.
 * Output: out[ b ][ head ][ t ][ dk ]
 *
 * @param in         Input tensor (B × T × in_stride).
 * @param out        Output tensor (B × nH × T × dK).
 * @param B          Batch size.
 * @param T          Sequence length.
 * @param nH         Number of attention heads.
 * @param dK         Per-head dimension.
 * @param in_stride  Row stride of the input tensor.  Use @c D (= nH*dK) for a
 *                   contiguous tensor, or @c 3*D when reading from an
 *                   interleaved QKV buffer.
 *
 * Total elements = B * T * nH * dK.
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

  // in:  [b, t, h*dK + dk]  with the given in_stride per token
  // out: [b, h, t, dk]
  int in_idx = b * (T * in_stride) + t * in_stride + h * dK + dk;
  int out_idx = ((b * nH + h) * T + t) * dK + dk;
  out[out_idx] = in[in_idx];
}

// =============================================================================
// Part 3 — Softmax (row‑wise, in‑place)
// =============================================================================

/**
 * @brief Row‑wise softmax for attention scores.
 *
 * data has shape (rows × cols).  Each block handles one row.
 * Uses the numerically stable log‑sum‑exp trick.
 *
 * @param data  Attention score matrix (rows × cols), normalized in place.
 * @param rows  Number of rows.
 * @param cols  Number of columns per row.
 */
__global__ void softmax_kernel(float* __restrict__ data, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;

  extern __shared__ float smem[];
  float* row_data = data + row * cols;

  // Find max
  float local_max = kNegInf;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, row_data[c]);
  smem[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float max_val = smem[0];

  // Exp and sum
  float local_sum = 0.0F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf(row_data[c] - max_val);
    row_data[c] = e;
    local_sum += e;
  }
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float total = smem[0];

  // Normalize
  for (int c = threadIdx.x; c < cols; c += blockDim.x) row_data[c] /= total;
}

// =============================================================================
// Part 4 — Merge heads: (B, nH, T, dK) → (B, T, nH*dK)
// =============================================================================

/**
 * @brief Merge heads: transpose (B, nH, T, dK) → (B, T, nH*dK).
 *
 * Inverse of split_heads_kernel — reassembles per-head outputs into a
 * single concatenated tensor for the output projection.
 *
 * @param in   Input tensor (B × nH × T × dK).
 * @param out  Output tensor (B × T × nH*dK).
 * @param B    Batch size.
 * @param T    Sequence length.
 * @param nH   Number of attention heads.
 * @param dK   Per-head dimension.
 */
__global__ void merge_heads_kernel(const float* __restrict__ in, float* __restrict__ out, int B,
                                   int T, int nH, int dK) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * T * nH * dK;
  if (idx >= total) return;

  int dk = idx % dK;
  int h = (idx / dK) % nH;
  int t = (idx / (dK * nH)) % T;
  int b = idx / (dK * nH * T);

  int in_idx = ((b * nH + h) * T + t) * dK + dk;
  int out_idx = b * (T * nH * dK) + t * (nH * dK) + h * dK + dk;
  out[out_idx] = in[in_idx];
}

// =============================================================================
// MultiHeadAttention struct
// =============================================================================

/**
 * @brief Encapsulates multi-head self-attention forward pass.
 *
 * Weight layout:
 *   W_QKV : (3 * D × D)  — projects input to [Q | K | V]
 *   W_O   : (D × D)      — output projection
 */
struct MultiHeadAttention {
  int B, T, D, nH, dK;
  cublasHandle_t handle;
  float* W_QKV;  // (3D × D)
  float* W_O;    // (D × D)

  // Scratch buffers (caller allocates)
  float* qkv_buf;  // (B × T × 3D)
  float* Q;        // (B × nH × T × dK)
  float* K;
  float* V;
  float* scores;   // (B × nH × T × T)
  float* context;  // (B × nH × T × dK)
  float* merged;   // (B × T × D)

  /**
   * @brief Runs the full multi-head self-attention forward pass.
   *
   * @param d_X    Input tensor (B × T × D), device pointer.
   * @param d_out  Output tensor (B × T × D), device pointer.
   */
  void forward(const float* d_X, float* d_out) {
    float alpha = 1.0F, beta = 0.0F;

    // Part 1: QKV projection  (B*T × D) @ W_QKV^T → (B*T × 3D)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3 * D, B * T, D, &alpha, W_QKV, D,
                             d_X, D, &beta, qkv_buf, 3 * D));

    // Part 2: Split into Q, K, V  — each (B, nH, T, dK)
    // qkv_buf has interleaved layout (B, T, 3D), so use stride 3*D.
    int total = B * T * D;
    int grid = (total + kBlockSize - 1) / kBlockSize;
    split_heads_kernel<<<grid, kBlockSize>>>(qkv_buf, Q, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    split_heads_kernel<<<grid, kBlockSize>>>(qkv_buf + D, K, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());
    split_heads_kernel<<<grid, kBlockSize>>>(qkv_buf + 2 * D, V, B, T, nH, dK, 3 * D);
    CUDA_CHECK(cudaGetLastError());

    // Part 3a: Scores = Q @ K^T  →  (B*nH × T × T)
    // Treat (B*nH) as batch.  Q is (T × dK), K is (T × dK).
    // Scores_ij = sum_k Q[i,k] * K[j,k]  → cublas: C = K^T @ Q with lda etc.
    float scale = 1.0F / sqrtf(static_cast<float>(dK));
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, dK, &scale, K,
                                           dK, T * dK, Q, dK, T * dK, &beta, scores, T, T * T,
                                           B * nH));

    // Part 3b: Softmax
    int smem_block = 128;
    softmax_kernel<<<B * nH * T, smem_block, smem_block * sizeof(float)>>>(scores, B * nH * T, T);
    CUDA_CHECK(cudaGetLastError());

    // Part 3c: Context = Attn @ V  →  (B*nH × T × dK)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dK, T, T, &alpha, V,
                                           dK, T * dK, scores, T, T * T, &beta, context, dK, T * dK,
                                           B * nH));

    // Part 4a: Merge heads
    merge_heads_kernel<<<grid, kBlockSize>>>(context, merged, B, T, nH, dK);
    CUDA_CHECK(cudaGetLastError());

    // Part 4b: Output projection  (B*T × D) @ W_O^T → (B*T × D)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, B * T, D, &alpha, W_O, D, merged,
                             D, &beta, d_out, D));
  }
};

// =============================================================================
// main
// =============================================================================

int main() {
  constexpr int kB = 2;
  constexpr int kT = 8;
  constexpr int kD = 64;
  constexpr int kNH = 4;
  constexpr int kDK = kD / kNH;  // 16

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---- Allocate weights ----
  float *d_W_QKV, *d_W_O;
  CUDA_CHECK(cudaMalloc(&d_W_QKV, 3 * kD * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_W_O, kD * kD * sizeof(float)));

  // Xavier-ish init
  std::vector<float> h_wqkv(3 * kD * kD);
  std::vector<float> h_wo(kD * kD);
  float fan = 1.0F / sqrtf(static_cast<float>(kD));
  for (auto& v : h_wqkv)
    v = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5F) * 2.0F * fan;
  for (auto& v : h_wo)
    v = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5F) * 2.0F * fan;
  CUDA_CHECK(
      cudaMemcpy(d_W_QKV, h_wqkv.data(), 3 * kD * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W_O, h_wo.data(), kD * kD * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Allocate input ----
  float* d_X;
  CUDA_CHECK(cudaMalloc(&d_X, kB * kT * kD * sizeof(float)));
  std::vector<float> h_X(kB * kT * kD);
  for (auto& v : h_X) v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.1F;
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), kB * kT * kD * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Allocate scratch ----
  float *d_qkv, *d_Q, *d_K, *d_V, *d_scores, *d_ctx, *d_merged, *d_out;
  CUDA_CHECK(cudaMalloc(&d_qkv, kB * kT * 3 * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Q, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scores, kB * kNH * kT * kT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ctx, kB * kNH * kT * kDK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_merged, kB * kT * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, kB * kT * kD * sizeof(float)));

  MultiHeadAttention mha{kB,    kT,  kD,  kNH, kDK,      handle, d_W_QKV, d_W_O,
                         d_qkv, d_Q, d_K, d_V, d_scores, d_ctx,  d_merged};

  std::printf("=== Multi‑Head Self‑Attention ===\n");
  std::printf("  B=%d T=%d D=%d nH=%d dK=%d\n", kB, kT, kD, kNH, kDK);
  mha.forward(d_X, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Print a few output values
  std::vector<float> h_out(kB * kT * kD);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kB * kT * kD * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  out[0] = %.6f\n", static_cast<double>(h_out[0]));
  std::printf("  out[D] = %.6f\n", static_cast<double>(h_out[kD]));

  // ---- Cleanup ----
  CUDA_CHECK(cudaFree(d_W_QKV));
  CUDA_CHECK(cudaFree(d_W_O));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_qkv));
  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_ctx));
  CUDA_CHECK(cudaFree(d_merged));
  CUDA_CHECK(cudaFree(d_out));
  CUBLAS_CHECK(cublasDestroy(handle));

  std::printf("\nDone.\n");
  return EXIT_SUCCESS;
}
