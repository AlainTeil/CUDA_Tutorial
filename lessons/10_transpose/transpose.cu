/**
 * @file transpose.cu
 * @brief Lesson 10 — Matrix Transpose (naive & shared-memory tiled).
 *
 * Transpose is a simple operation (`out[c][r] = in[r][c]`) but a challenging
 * one for GPUs because it fundamentally changes the memory access pattern.
 *
 * ## The coalescing problem
 *
 * In row-major layout, adjacent columns are adjacent in memory.
 * - **Reading** `in[row][col]` with consecutive threads varying `col` is
 *   coalesced (good).
 * - **Writing** `out[col][row]` with consecutive threads varying `col` means
 *   consecutive threads write to non-adjacent rows — **scattered** writes
 *   (bad: each write goes to a different cache line).
 *
 * ## The tiled solution
 *
 * 1. Load a TILE×TILE sub-matrix from `in` into shared memory (coalesced
 *    reads).  Shared memory has no coalescing requirement.
 * 2. Write the transposed tile from shared memory to `out`.  By swapping
 *    `blockIdx.x` and `blockIdx.y` for the output, consecutive threads
 *    again write to consecutive output addresses → **coalesced writes**.
 *
 * ## Bank conflicts
 *
 * Shared memory is divided into 32 **banks** (one per warp lane).  If two
 * threads in the same warp access the same bank (but different addresses),
 * the accesses are serialised (a "bank conflict").  The classic transpose
 * tile `tile[32][32]` causes 32-way bank conflicts because column 0 of
 * every row maps to bank 0.
 *
 * **Fix:** declare `tile[32][33]` (the "+1 padding trick").  The extra
 * column shifts each row by one bank, eliminating all conflicts.  Here we
 * use `tile[kTile][kTile + 1]`.
 */

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

constexpr int kTile = 32;

// =============================================================================
// Naive transpose
// =============================================================================

/// @brief Naive transpose: coalesced reads, scattered writes.
///
/// Each thread reads `in[row * cols + col]` (coalesced, since consecutive
/// threads have consecutive `col` values) and writes to
/// `out[col * rows + row]` (scattered, since consecutive threads write
/// to different columns of the output = different cache lines).
///
/// On a large matrix this achieves only ~50% of peak memory bandwidth
/// due to the uncoalesced writes.
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    out[col * rows + row] = in[row * cols + col];
  }
}

// =============================================================================
// Tiled transpose (shared memory, bank-conflict-free)
// =============================================================================

/// @brief Tiled transpose: coalesced reads AND writes, no bank conflicts.
///
/// ### Step 1 — Load tile (coalesced read)
/// Consecutive threads read consecutive columns of `in` into
/// `tile[threadIdx.y][threadIdx.x]`.  This is coalesced.
///
/// ### Step 2 — Write transposed tile (coalesced write)
/// We compute the output position by swapping `blockIdx.x` and `blockIdx.y`.
/// Then we write `tile[threadIdx.x][threadIdx.y]` — note the swapped
/// indices, which performs the transpose.  Consecutive threads have
/// consecutive `threadIdx.x`, so they read consecutive shared-memory
/// columns and write consecutive output addresses → coalesced.
///
/// The `kTile + 1` padding on the shared memory array prevents bank
/// conflicts when reading the transposed access pattern
/// `tile[threadIdx.x][threadIdx.y]`.
__global__ void transpose_tiled(const float* in, float* out, int rows, int cols) {
  // +1 column to avoid bank conflicts on 32-wide shared memory.
  __shared__ float tile[kTile][kTile + 1];

  int col = blockIdx.x * kTile + threadIdx.x;
  int row = blockIdx.y * kTile + threadIdx.y;

  // Coalesced read from global memory into shared memory
  if (row < rows && col < cols) {
    tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
  }
  __syncthreads();

  // Write transposed tile — swap blockIdx.x/y to ensure coalesced writes
  int out_col = blockIdx.y * kTile + threadIdx.x;
  int out_row = blockIdx.x * kTile + threadIdx.y;

  if (out_row < cols && out_col < rows) {
    out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
  }
}

// =============================================================================
// main
// =============================================================================
int main() {
  constexpr int kRows = 1024;
  constexpr int kCols = 512;
  size_t in_bytes = static_cast<size_t>(kRows) * kCols * sizeof(float);
  size_t out_bytes = static_cast<size_t>(kCols) * kRows * sizeof(float);

  std::vector<float> h_in(static_cast<size_t>(kRows) * kCols);
  for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice));

  dim3 threads(kTile, kTile);
  dim3 blocks((kCols + kTile - 1) / kTile, (kRows + kTile - 1) / kTile);

  // Naive
  transpose_naive<<<blocks, threads>>>(d_in, d_out, kRows, kCols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_naive(static_cast<size_t>(kCols) * kRows);
  CUDA_CHECK(cudaMemcpy(h_naive.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  // Tiled
  transpose_tiled<<<blocks, threads>>>(d_in, d_out, kRows, kCols);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_tiled(static_cast<size_t>(kCols) * kRows);
  CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  // Verify: both should equal CPU transpose
  bool ok = true;
  for (int r = 0; r < kRows && ok; ++r) {
    for (int c = 0; c < kCols && ok; ++c) {
      float expected = h_in[static_cast<size_t>(r) * kCols + c];
      float naive_val = h_naive[static_cast<size_t>(c) * kRows + r];
      float tiled_val = h_tiled[static_cast<size_t>(c) * kRows + r];
      if (naive_val != expected || tiled_val != expected) ok = false;
    }
  }
  std::printf("Transpose: %s\n", ok ? "PASSED" : "FAILED");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
