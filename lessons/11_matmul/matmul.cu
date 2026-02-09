/**
 * @file matmul.cu
 * @brief Lesson 11 — Matrix Multiplication (naive & shared-memory tiled).
 *
 * Matrix multiplication (GEMM) is the most important kernel in deep learning.
 * Fully connected layers, attention, convolutions (via im2col) — they all
 * reduce to GEMM.  Understanding how to optimise it on a GPU is essential.
 *
 * ## Arithmetic intensity
 *
 * For C = A(M×K) × B(K×N):
 *   - FLOPs: 2·M·N·K  (one multiply + one add per element of C, for each k)
 *   - Data:  M·K + K·N + M·N  floats  (read A, read B, write C)
 *   - AI ≈ 2·K for large square matrices — very high, so GEMM is
 *     **compute-bound** (unlike most kernels which are memory-bound).
 *
 * ## Naive kernel
 *
 * Each thread computes one element of C by reading an entire row of A and
 * column of B.  This is O(K) global memory reads per thread, and many of
 * those reads are **redundant**: thread (r, c) and thread (r, c+1) both
 * read the same row of A.
 *
 * ## Tiled kernel
 *
 * We partition A and B into TILE×TILE sub-matrices.  In each iteration:
 *   1. All threads cooperatively load one tile of A and one tile of B into
 *      shared memory (→ coalesced reads from global memory).
 *   2. Each thread computes a partial dot product using the shared tiles.
 *   3. `__syncthreads()` before the next tile overwrites shared memory.
 *
 * This reduces global memory reads by a factor of TILE (from O(K) to
 * O(K/TILE) per thread), and the shared-memory reads are fast and
 * bank-conflict-free.  With TILE = 16, that's a 16× reduction in global
 * memory traffic.
 *
 * ## Beyond this lesson
 *
 * Production GEMM kernels (cuBLAS, CUTLASS) add register tiling, double
 * buffering, warp-level matrix operations (WMMA / Tensor Cores), and
 * software pipelining.  Those achieve >90% of peak FLOPS.  See Lessons 19
 * (cuBLAS) and 21 (Tensor Cores) for library-based approaches.
 */

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

constexpr int kTile = 16;

// =============================================================================
// Naive matmul
// =============================================================================

/// @brief Naive matmul: each thread computes one element of C.
///
/// Thread (row, col) reads all K elements of A[row, :] and B[:, col] from
/// global memory.  This results in O(M*N*K) total global reads across all
/// threads, with massive redundancy — the same row of A is read by N threads.
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float sum = 0.0F;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// =============================================================================
// Tiled matmul (shared memory)
// =============================================================================

/// @brief Tiled matmul: shared-memory tiles reduce global reads by TILE×.
///
/// The outer loop iterates over K in steps of `kTile`.  In each step:
///   1. Each thread loads one element of the A-tile and one of the B-tile
///      from global memory into shared memory (coalesced).
///   2. `__syncthreads()` ensures the tiles are fully populated.
///   3. Each thread accumulates `kTile` multiply-adds from shared memory.
///   4. A second `__syncthreads()` before the next iteration prevents the
///      new tile load from overwriting data still being read.
///
/// This transforms O(K) global reads per thread into O(K/kTile) global
/// reads + O(K) shared-memory reads (which are ~100× faster).
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  int col = blockIdx.x * kTile + threadIdx.x;
  int row = blockIdx.y * kTile + threadIdx.y;

  float sum = 0.0F;

  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    // Load A tile
    int a_col = t * kTile + threadIdx.x;
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0F;

    // Load B tile
    int b_row = t * kTile + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0F;

    __syncthreads();

    for (int k = 0; k < kTile; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// =============================================================================
// CPU reference
// =============================================================================

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float sum = 0.0F;
      for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

int main() {
  constexpr int M = 256, N = 256, K = 128;
  size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
  size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
  size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N),
      hC_ref(static_cast<size_t>(M) * N);

  for (auto& v : hA) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));
  for (auto& v : hB) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));

  cpu_matmul(hA.data(), hB.data(), hC_ref.data(), M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, a_bytes));
  CUDA_CHECK(cudaMalloc(&dB, b_bytes));
  CUDA_CHECK(cudaMalloc(&dC, c_bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

  // Naive
  matmul_naive<<<blocks, threads>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC_naive(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hC_naive.data(), dC, c_bytes, cudaMemcpyDeviceToHost));

  // Tiled
  matmul_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hC_tiled(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hC_tiled.data(), dC, c_bytes, cudaMemcpyDeviceToHost));

  float max_err_naive = 0, max_err_tiled = 0;
  for (size_t i = 0; i < hC_ref.size(); ++i) {
    max_err_naive = std::max(max_err_naive, std::abs(hC_naive[i] - hC_ref[i]));
    max_err_tiled = std::max(max_err_tiled, std::abs(hC_tiled[i] - hC_ref[i]));
  }

  std::printf("Matmul (%dx%d × %dx%d):\n", M, K, K, N);
  std::printf("  Naive max error : %e\n", static_cast<double>(max_err_naive));
  std::printf("  Tiled max error : %e\n", static_cast<double>(max_err_tiled));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return EXIT_SUCCESS;
}
