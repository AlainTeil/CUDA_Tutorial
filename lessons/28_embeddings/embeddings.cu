/**
 * @file embeddings.cu
 * @brief Lesson 28 — Token Embeddings & Positional Encoding.
 *
 * Every NLP (and many vision) model begins by converting discrete token IDs
 * into dense vectors.  This lesson implements:
 *
 * ## Part 1 — Token Embedding Lookup (Forward)
 *
 * Given a vocabulary of V tokens, each mapped to a D‑dimensional vector, the
 * **embedding table** is a (V × D) matrix `W`.  For a batch of B sequences,
 * each of length T, the forward pass is:
 *
 *     out[b][t][d] = W[ ids[b][t] ][ d ]
 *
 * Each thread handles one `(b*T + t, d)` element — pure gather.
 *
 * ## Part 2 — Sinusoidal Positional Encoding
 *
 * Transformers lack built‑in position awareness.  Following Vaswani et al.
 * (2017), we add a fixed positional signal:
 *
 *     PE(pos, 2i)   = sin( pos / 10000^{2i/D} )
 *     PE(pos, 2i+1) = cos( pos / 10000^{2i/D} )
 *
 * A kernel computes PE on the fly and adds it element‑wise to the embedding
 * output.
 *
 * ## Part 3 — Embedding Backward (Gradient Scatter)
 *
 * Because multiple positions may map to the same token, the backward pass
 * uses `atomicAdd` to accumulate gradients:
 *
 *     dW[ids[b][t]][d] += dout[b][t][d]
 *
 * ## Part 4 — cuBLAS Matmul for Output Projection
 *
 * After embedding, it is common to project the D‑dimensional vectors through
 * a linear layer.  We demonstrate cuBLAS `cublasSgemm` for this step (see
 * Lesson 19 for a cuBLAS primer).
 *
 * See Lesson 31 for the self‑attention mechanism that consumes these
 * embeddings, and Lesson 32 for the full Transformer encoder.
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

/// Base frequency for sinusoidal positional encoding.
constexpr float kPEBase = 10000.0F;

// =============================================================================
// Part 1 — Token Embedding Forward
// =============================================================================

/**
 * @brief Gather rows from the embedding table.
 * @param table  Embedding table (V × D), row‑major
 * @param ids    Token ID array (B*T)
 * @param out    Output (B*T × D), row‑major
 * @param total  B * T
 * @param D      Embedding dimension
 * @param V      Vocabulary size (for bounds check)
 */
__global__ void embedding_forward_kernel(const float* __restrict__ table,
                                         const int* __restrict__ ids, float* __restrict__ out,
                                         int total, int D, int V) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;

  int row = idx / D;
  int col = idx % D;
  int token_id = ids[row];
  assert(token_id >= 0 && token_id < V && "token ID out of range");
  out[row * D + col] = table[token_id * D + col];
}

// =============================================================================
// Part 2 — Sinusoidal Positional Encoding
// =============================================================================

/**
 * @brief Add sinusoidal positional encoding in‑place.
 * @param data   (B*T × D) embedding output — modified in place
 * @param T      Sequence length
 * @param D      Embedding dimension
 * @param total  B * T
 *
 * Thread maps to (row, col).  Position within the sequence is `row % T`.
 */
__global__ void sinusoidal_pe_kernel(float* __restrict__ data, int T, int D, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;

  int row = idx / D;
  int col = idx % D;
  int pos = row % T;

  float exponent = static_cast<float>(col / 2 * 2) / static_cast<float>(D);
  float freq = 1.0F / powf(kPEBase, exponent);
  float angle = static_cast<float>(pos) * freq;

  float pe = (col % 2 == 0) ? sinf(angle) : cosf(angle);
  data[idx] += pe;
}

// =============================================================================
// Part 3 — Embedding Backward (Gradient Scatter)
// =============================================================================

/**
 * @brief Scatter‑add gradients back into the embedding table gradient.
 * @param d_table_grad  (V × D) — accumulated via atomicAdd
 * @param ids           Token ID array (B*T)
 * @param dout          Upstream gradient (B*T × D)
 * @param total         B * T
 * @param D             Embedding dimension
 */
__global__ void embedding_backward_kernel(float* __restrict__ d_table_grad,
                                          const int* __restrict__ ids,
                                          const float* __restrict__ dout, int total, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total * D) return;

  int row = idx / D;
  int col = idx % D;
  int token_id = ids[row];
  atomicAdd(&d_table_grad[token_id * D + col], dout[row * D + col]);
}

// =============================================================================
// Part 4 — Linear projection via cuBLAS
// =============================================================================

/**
 * @brief project = X @ W^T  where X is (M × K) and W is (N × K).
 *
 * cuBLAS is column‑major, so we compute  C = W * X^T  → C^T = X * W^T.
 * We pass row‑major matrices by swapping A/B and transposing.
 */
static void linear_project(cublasHandle_t handle, const float* d_X, const float* d_W, float* d_out,
                           int M, int K, int N) {
  float alpha = 1.0F, beta = 0.0F;
  // C(M,N) = X(M,K) * W^T(K,N)  — stored row‑major
  // cuBLAS:  C_col = W * X^T      with lda=N, ldb=K, ldc=N
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_W, K, d_X, K, &beta,
                           d_out, N));
}

// =============================================================================
// main
// =============================================================================

int main() {
  // Hyper‑parameters
  constexpr int kV = 256;  // Vocabulary size
  constexpr int kD = 64;   // Embedding dimension
  constexpr int kB = 4;    // Batch size
  constexpr int kT = 16;   // Sequence length
  constexpr int kTotal = kB * kT;
  constexpr int kProjDim = 32;  // Output projection dimension

  // ---- Host data ----
  std::vector<float> h_table(kV * kD);
  for (int i = 0; i < static_cast<int>(h_table.size()); ++i)
    h_table[i] = static_cast<float>(i % 7) * 0.01F;

  std::vector<int> h_ids(kTotal);
  for (int i = 0; i < kTotal; ++i) h_ids[i] = i % kV;

  std::vector<float> h_proj_w(kProjDim * kD, 0.01F);

  // ---- Device allocations ----
  float *d_table, *d_out, *d_table_grad, *d_dout, *d_proj_w, *d_proj_out;
  int* d_ids;

  CUDA_CHECK(cudaMalloc(&d_table, kV * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, kTotal * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_table_grad, kV * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dout, kTotal * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ids, kTotal * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_proj_w, kProjDim * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_proj_out, kTotal * kProjDim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), kV * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), kTotal * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_proj_w, h_proj_w.data(), kProjDim * kD * sizeof(float), cudaMemcpyHostToDevice));

  int elems = kTotal * kD;
  int grid = (elems + kBlockSize - 1) / kBlockSize;

  // Part 1: Forward
  std::printf("=== Part 1: Embedding Forward ===\n");
  embedding_forward_kernel<<<grid, kBlockSize>>>(d_table, d_ids, d_out, kTotal, kD, kV);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Part 2: Add positional encoding
  std::printf("=== Part 2: Sinusoidal Positional Encoding ===\n");
  sinusoidal_pe_kernel<<<grid, kBlockSize>>>(d_out, kT, kD, kTotal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float first = 0.0F;
  CUDA_CHECK(cudaMemcpy(&first, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  out[0] = %.6f\n", static_cast<double>(first));

  // Part 3: Backward
  std::printf("=== Part 3: Embedding Backward ===\n");
  // Use ones as upstream gradient
  std::vector<float> h_dout(kTotal * kD, 1.0F);
  CUDA_CHECK(
      cudaMemcpy(d_dout, h_dout.data(), kTotal * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_table_grad, 0, kV * kD * sizeof(float)));

  embedding_backward_kernel<<<grid, kBlockSize>>>(d_table_grad, d_ids, d_dout, kTotal, kD);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float grad0 = 0.0F;
  CUDA_CHECK(cudaMemcpy(&grad0, d_table_grad, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  d_table_grad[0] = %.4f\n", static_cast<double>(grad0));

  // Part 4: cuBLAS projection
  std::printf("=== Part 4: cuBLAS Output Projection ===\n");
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  linear_project(handle, d_out, d_proj_w, d_proj_out, kTotal, kD, kProjDim);
  CUDA_CHECK(cudaDeviceSynchronize());

  float proj0 = 0.0F;
  CUDA_CHECK(cudaMemcpy(&proj0, d_proj_out, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  proj_out[0] = %.6f\n", static_cast<double>(proj0));

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_table_grad));
  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_ids));
  CUDA_CHECK(cudaFree(d_proj_w));
  CUDA_CHECK(cudaFree(d_proj_out));

  std::printf("\nDone.\n");
  return EXIT_SUCCESS;
}
