/**
 * @file thread_hierarchy.cu
 * @brief Lesson 05 — CUDA Thread Hierarchy: 1D, 2D, and 3D grids.
 *
 * CUDA organises threads into a two-level hierarchy:
 *
 * ```
 * Grid
 *   └─ Block (0,0)   Block (1,0)   Block (2,0)  ...
 *        └─ Thread (0,0)  Thread (1,0)  Thread (2,0) ...
 *        └─ Thread (0,1)  Thread (1,1)  ...
 * ```
 *
 * - **Grid** — the collection of all blocks launched by a single kernel.
 *   Dimensions: up to 3-D (`gridDim.x`, `.y`, `.z`).
 * - **Block** — a group of threads that can cooperate via shared memory and
 *   `__syncthreads()`.  Dimensions: up to 3-D (`blockDim.x`, `.y`, `.z`).
 *   Max threads per block is typically 1024.
 *
 * ## Why multi-dimensional indexing?
 *
 * For 1-D arrays, a 1-D grid is natural.  But image processing and matrix
 * operations map naturally to 2-D grids (row, col), and volume processing
 * (CT scans, 3-D simulations) maps to 3-D grids (x, y, z).  Using matching
 * grid dimensions makes the indexing code clearer and helps the hardware
 * scheduler map threads to data efficiently.
 *
 * ## Global index formulas
 *
 * - **1-D**: `idx = blockIdx.x * blockDim.x + threadIdx.x`
 * - **2-D**: `row = blockIdx.y * blockDim.y + threadIdx.y`
 *           `col = blockIdx.x * blockDim.x + threadIdx.x`
 * - **3-D**: add `z = blockIdx.z * blockDim.z + threadIdx.z`
 *
 * This lesson creates output arrays using 1-D, 2-D, and 3-D grids and
 * verifies that every element is filled with its linearised index.
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

// =============================================================================
// 1-D kernel
// =============================================================================

/**
 * @brief Each thread writes its 1-D global index.
 *
 * The simplest case: index = blockIdx.x * blockDim.x + threadIdx.x.
 * All lessons in Phase 1 and 2 use this formula for 1-D kernels.
 */
__global__ void fill_1d(int* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = idx;
}

// =============================================================================
// 2-D kernel — fill a rows×cols matrix
// =============================================================================

/**
 * @brief Each thread writes its (row, col) linearized index into a 2-D matrix.
 *
 * Grid and block dimensions are both `dim3` (a CUDA struct with `.x`, `.y`,
 * `.z` fields).  For 2-D indexing, `.x` maps to columns and `.y` to rows
 * (matching the memory layout where consecutive column indices are adjacent
 * in memory for row-major storage).
 *
 * @param out   Device pointer to row-major matrix of size rows * cols.
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 */
__global__ void fill_2d(int* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    out[row * cols + col] = row * cols + col;
  }
}

// =============================================================================
// 3-D kernel — fill a depth × rows × cols volume
// =============================================================================

/**
 * @brief 3-D indexing: each thread writes its (z, y, x) linearized index.
 *
 * With a 3-D grid, every thread has a unique (x, y, z) coordinate.
 * The linearised index is `z * rows * cols + y * cols + x`, mirroring
 * how a 3-D C array `a[depth][rows][cols]` is laid out in memory.
 */
__global__ void fill_3d(int* out, int depth, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < cols && y < rows && z < depth) {
    out[z * rows * cols + y * cols + x] = z * rows * cols + y * cols + x;
  }
}

// =============================================================================
// main
// =============================================================================
int main() {
  // ---- 1-D ------------------------------------------------------------------
  {
    constexpr int kN = 1000;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(kN) * sizeof(int)));
    fill_1d<<<(kN + 255) / 256, 256>>>(d_out, kN);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(kN);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, kN * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kN && ok; ++i) ok = (h[static_cast<size_t>(i)] == i);
    std::printf("1-D fill: %s\n", ok ? "PASSED" : "FAILED");
    CUDA_CHECK(cudaFree(d_out));
  }

  // ---- 2-D ------------------------------------------------------------------
  {
    constexpr int kRows = 64, kCols = 48;
    constexpr int kTotal = kRows * kCols;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(kTotal) * sizeof(int)));

    dim3 threads(16, 16);
    dim3 blocks((kCols + 15) / 16, (kRows + 15) / 16);
    fill_2d<<<blocks, threads>>>(d_out, kRows, kCols);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(kTotal);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, kTotal * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kTotal && ok; ++i) ok = (h[static_cast<size_t>(i)] == i);
    std::printf("2-D fill (%dx%d): %s\n", kRows, kCols, ok ? "PASSED" : "FAILED");
    CUDA_CHECK(cudaFree(d_out));
  }

  // ---- 3-D ------------------------------------------------------------------
  {
    constexpr int kD = 4, kR = 8, kC = 16;
    constexpr int kTotal = kD * kR * kC;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(kTotal) * sizeof(int)));

    dim3 threads(8, 4, 2);
    dim3 blocks((kC + 7) / 8, (kR + 3) / 4, (kD + 1) / 2);
    fill_3d<<<blocks, threads>>>(d_out, kD, kR, kC);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(kTotal);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, kTotal * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kTotal && ok; ++i) ok = (h[static_cast<size_t>(i)] == i);
    std::printf("3-D fill (%dx%dx%d): %s\n", kD, kR, kC, ok ? "PASSED" : "FAILED");
    CUDA_CHECK(cudaFree(d_out));
  }

  return EXIT_SUCCESS;
}
