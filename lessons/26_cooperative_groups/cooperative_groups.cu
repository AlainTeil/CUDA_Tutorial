/**
 * @file cooperative_groups.cu
 * @brief Lesson 26 — Cooperative Groups.
 *
 * CUDA's **Cooperative Groups** API (introduced in CUDA 9) provides a
 * *type-safe*, *composable* abstraction for thread synchronisation that
 * replaces raw `__syncthreads()` and `__shfl_down_sync(0xFFFFFFFF, ...)`.
 *
 * ## Synchronisation hierarchy
 *
 * ```
 *   grid_group              ← entire grid (requires cooperative launch)
 *       │
 *   thread_block            ← replaces __syncthreads()
 *       │
 *   thread_block_tile<N>    ← static sub-warp partition (N = 1..32, power of 2)
 *       │
 *   coalesced_threads()     ← dynamically formed group of active threads
 * ```
 *
 * ## Part 1 — Block-level partitioning
 *
 * | Old API                               | CG equivalent                          |
 * |---------------------------------------|----------------------------------------|
 * | `__syncthreads()`                     | `cg::thread_block::sync()`             |
 * | `__shfl_down_sync(mask, v, d)`        | `tile.shfl_down(v, d)`                 |
 * | manual warp-level reduce              | `cg::reduce(tile, v, cg::plus<>())`   |
 * | `if (threadIdx.x < 32) { ... }`       | `auto tile = cg::tiled_partition<32>(block)` |
 *
 * The key advantage: the CG API **cannot** produce incorrect mask bugs
 * (e.g. passing `0xFFFFFFFF` when not all warp lanes are active).
 *
 * ## Part 2 — Grid-level cooperative launch
 *
 * `cg::grid_group` enables **grid-wide synchronisation** via
 * `grid.sync()`.  This is impossible with raw CUDA APIs — normally
 * blocks cannot synchronise with each other.  The constraint is that the
 * grid must fit into the GPU's active block capacity, so the grid size is
 * limited to what `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports.
 *
 * This enables **single-pass global reduction**: every block reduces
 * locally → `grid.sync()` → block 0 combines the partial results — all
 * in one kernel launch, without recursive launches or atomics.
 *
 * See Lesson 08 for the original reduction patterns this lesson modernises.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (call);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Part 1a — Warp-tile reduction using cooperative groups
// =============================================================================

/**
 * @brief Reduction using `thread_block_tile<32>` (warp-level CG).
 *
 * Contrast with Lesson 08's `__shfl_down_sync(0xFFFFFFFF, val, offset)`.
 * Here the tile abstraction guarantees correct synchronisation without
 * explicit mask management.
 *
 * @param in   Input array of floats.
 * @param out  Output scalar (atomically accumulated across blocks).
 * @param n    Number of input elements.
 */
__global__ void reduce_tile_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread loads one element (or 0 if out of bounds)
  float val = (idx < n) ? in[idx] : 0.0F;

  // Warp-level reduction using CG
  for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
    val += tile.shfl_down(val, offset);
  }

  // Lane 0 of each warp writes partial sum to shared memory
  __shared__ float warp_sums[kBlockSize / 32];
  if (tile.thread_rank() == 0) {
    warp_sums[threadIdx.x / 32] = val;
  }
  block.sync();

  // First warp reduces the warp sums
  int num_warps = blockDim.x / 32;
  if (threadIdx.x < static_cast<unsigned>(num_warps)) {
    val = warp_sums[threadIdx.x];
    auto final_tile = cg::tiled_partition<32>(block);
    for (int offset = final_tile.size() / 2; offset > 0; offset /= 2) {
      val += final_tile.shfl_down(val, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(out, val);
    }
  }
}

// =============================================================================
// Part 1b — One-liner reduction using cg::reduce
// =============================================================================

/**
 * @brief CG `reduce()` — the simplest possible reduction.
 *
 * `cg::reduce(tile, val, cg::plus<float>())` performs a tree reduction
 * within the tile using shuffle intrinsics internally.  This single line
 * replaces ~10 lines of manual __shfl_down_sync code.
 *
 * @param in   Input array of floats.
 * @param out  Output scalar (atomically accumulated across blocks).
 * @param n    Number of input elements.
 */
__global__ void reduce_cg_builtin_kernel(const float* __restrict__ in, float* __restrict__ out,
                                         int n) {
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < n) ? in[idx] : 0.0F;

  // One-liner warp reduction
  float warp_sum = cg::reduce(tile, val, cg::plus<float>());

  __shared__ float warp_sums[kBlockSize / 32];
  if (tile.thread_rank() == 0) warp_sums[threadIdx.x / 32] = warp_sum;
  block.sync();

  if (threadIdx.x < static_cast<unsigned>(blockDim.x / 32)) {
    float v = warp_sums[threadIdx.x];
    auto ftile = cg::tiled_partition<32>(block);
    float block_sum = cg::reduce(ftile, v, cg::plus<float>());
    if (threadIdx.x == 0) atomicAdd(out, block_sum);
  }
}

// =============================================================================
// Part 1c — Coalesced threads (dynamic groups after divergent branch)
// =============================================================================

/**
 * @brief Demonstrates `cg::coalesced_threads()` — a dynamic group of
 *        only the **active** threads at a divergent branch point.
 *
 * Counts how many elements satisfy a predicate using coalesced groups.
 * Only threads with qualifying elements form the group and participate
 * in the reduction.
 *
 * @param in         Input array of floats.
 * @param count      Output element count (atomically incremented).
 * @param n          Number of input elements.
 * @param threshold  Elements above this value are counted.
 */
__global__ void coalesced_filter_kernel(const float* __restrict__ in, int* __restrict__ count,
                                        int n, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  if (in[idx] > threshold) {
    auto active = cg::coalesced_threads();
    // active.size() = number of threads in this warp that passed the branch
    if (active.thread_rank() == 0) {
      atomicAdd(count, static_cast<int>(active.size()));
    }
  }
}

// =============================================================================
// Part 2 — Grid-level cooperative reduction (single-pass)
// =============================================================================

/**
 * @brief Single-pass global reduction using `cg::grid_group::sync()`.
 *
 * Algorithm:
 *   1. Each block reduces its portion → partial_sums[blockIdx.x]
 *   2. grid.sync()  — all blocks wait for each other
 *   3. Block 0 reduces partial_sums → final result
 *
 * This avoids the recursive kernel-launch approach of Lesson 08.
 *
 * @param in            Input array of floats.
 * @param partial_sums  Workspace for per-block partial sums (gridDim.x elements).
 * @param out           Output scalar — the final reduced sum.
 * @param n             Number of input elements.
 */
__global__ void grid_reduce_kernel(const float* __restrict__ in, float* __restrict__ partial_sums,
                                   float* __restrict__ out, int n) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float val = 0.0F;
  for (int i = idx; i < n; i += stride) val += in[i];

  // Block-level reduction
  float warp_sum = cg::reduce(tile, val, cg::plus<float>());
  __shared__ float warp_sums[kBlockSize / 32];
  if (tile.thread_rank() == 0) warp_sums[threadIdx.x / 32] = warp_sum;
  block.sync();

  float block_sum = 0.0F;
  int n_warps = blockDim.x / 32;
  if (threadIdx.x < 32u) {
    float lane_val = (threadIdx.x < static_cast<unsigned>(n_warps)) ? warp_sums[threadIdx.x] : 0.0F;
    block_sum = cg::reduce(tile, lane_val, cg::plus<float>());
  }
  if (threadIdx.x == 0) partial_sums[blockIdx.x] = block_sum;

  // Grid-wide barrier — all blocks must complete before block 0 merges
  grid.sync();

  // Block 0 reduces the partial sums
  if (blockIdx.x == 0) {
    float pval = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(gridDim.x);
         i += static_cast<int>(blockDim.x)) {
      pval += partial_sums[i];
    }
    // Full block reduction of pval: first warp sums, then tile reduce
    warp_sum = cg::reduce(tile, pval, cg::plus<float>());
    if (tile.thread_rank() == 0) warp_sums[threadIdx.x / 32] = warp_sum;
    block.sync();
    float total = 0.0F;
    if (threadIdx.x < 32u) {
      float lv = (threadIdx.x < static_cast<unsigned>(n_warps)) ? warp_sums[threadIdx.x] : 0.0F;
      total = cg::reduce(tile, lv, cg::plus<float>());
    }
    if (threadIdx.x == 0) *out = total;
  }
}

// =============================================================================
// main
// =============================================================================

int main() {
  constexpr int kN = 1 << 20;  // 1 M elements
  std::vector<float> h_in(kN);
  for (int i = 0; i < kN; ++i) h_in[static_cast<size_t>(i)] = 1.0F;  // sum = kN

  float* d_in;
  CUDA_CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));

  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  float expected = static_cast<float>(kN);

  // ---- Part 1a: tile reduction ----
  {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int grid = (kN + kBlockSize - 1) / kBlockSize;
    reduce_tile_kernel<<<grid, kBlockSize>>>(d_in, d_out, kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float result = 0.0F;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("Tile reduction  : %.0f (expected %.0f)\n", static_cast<double>(result),
                static_cast<double>(expected));
  }

  // ---- Part 1b: CG builtin reduce ----
  {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int grid = (kN + kBlockSize - 1) / kBlockSize;
    reduce_cg_builtin_kernel<<<grid, kBlockSize>>>(d_in, d_out, kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float result = 0.0F;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("CG reduce()     : %.0f (expected %.0f)\n", static_cast<double>(result),
                static_cast<double>(expected));
  }

  // ---- Part 1c: coalesced filter ----
  {
    // Count how many elements > 0.5  (all elements = 1.0, so count == kN)
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    int grid = (kN + kBlockSize - 1) / kBlockSize;
    coalesced_filter_kernel<<<grid, kBlockSize>>>(d_in, d_count, kN, 0.5F);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("Coalesced count : %d (expected %d)\n", count, kN);
    CUDA_CHECK(cudaFree(d_count));
  }

  // ---- Part 2: grid-level cooperative reduction ----
  {
    // Query max active blocks for cooperative launch
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, grid_reduce_kernel,
                                                             kBlockSize, 0));
    int max_grid = num_sms * max_blocks_per_sm;

    int need = (kN + kBlockSize - 1) / kBlockSize;
    int grid_size = (need < max_grid) ? need : max_grid;

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Cooperative launch
    void* args[] = {&d_in, &d_partial, &d_out,
                    const_cast<int*>(&kN)};  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(grid_reduce_kernel),
                                           dim3(grid_size), dim3(kBlockSize), args));
    CUDA_CHECK(cudaDeviceSynchronize());

    float result = 0.0F;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("Grid reduction  : %.0f (expected %.0f)  [%d blocks, limited to %d]\n",
                static_cast<double>(result), static_cast<double>(expected), grid_size, max_grid);
    CUDA_CHECK(cudaFree(d_partial));
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  return EXIT_SUCCESS;
}
