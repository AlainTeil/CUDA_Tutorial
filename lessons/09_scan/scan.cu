/**
 * @file scan.cu
 * @brief Lesson 09 — Prefix Sum (Exclusive Scan) using Blelloch algorithm.
 *
 * **Prefix sum** (scan) is the second fundamental parallel primitive after
 * reduction.  Given an array `[a0, a1, a2, ...]`, the exclusive prefix sum
 * produces `[0, a0, a0+a1, a0+a1+a2, ...]`.
 *
 * ## Why it matters
 *
 * Scan appears everywhere in parallel computing:
 *   - **Stream compaction** — filter an array in parallel (keep only passing
 *     elements, pack them contiguously).
 *   - **Radix sort** — counting sort at each digit uses scan.
 *   - **Sparse matrix operations** — building CSR row pointers.
 *   - **Cumulative histograms**, running totals, etc.
 *
 * ## The Blelloch algorithm (work-efficient scan)
 *
 * Two phases, both in-place on shared memory:
 *
 * 1. **Up-sweep (reduce)** — compute partial sums in a tree, exactly like
 *    the reduction in Lesson 08.  After this, `temp[n-1]` holds the total.
 *
 * 2. **Down-sweep** — Set the last element to the identity (0 for sum),
 *    then walk the tree back down, distributing partial sums so that each
 *    position holds its exclusive prefix sum.
 *
 * Both phases take O(n) work and O(log n) steps — this matches the
 * sequential algorithm’s O(n) work, hence "work-efficient".
 *
 * ## Multi-block extension
 *
 * Each block can scan up to `kBlockSize` elements.  For larger arrays:
 *   1. Scan each block independently, saving the block total.
 *   2. Scan the array of block totals (recursively).
 *   3. Add each scanned block total back to the corresponding block.
 *
 * This three-level strategy extends to arbitrary sizes.  In production,
 * use CUB’s `DeviceScan::ExclusiveSum` for a highly tuned implementation.
 */

#include <cstdio>
#include <cstdlib>
#include <numeric>
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

constexpr int kBlockSize = 256;

// =============================================================================
// Blelloch exclusive scan — single block
// =============================================================================

/**
 * @brief Blelloch exclusive scan within a single block.
 *
 * Each block operates on `blockDim.x` elements in shared memory.
 *
 * ### Up-sweep phase
 * Pairs of elements are added together with increasing stride (1, 2, 4, ...),
 * building a reduction tree.  The index formula `(tid+1)*stride*2 - 1`
 * selects the right child at each level.  After the up-sweep,
 * `temp[kBlockSize-1]` holds the total sum of the block.
 *
 * ### Down-sweep phase
 * We set `temp[kBlockSize-1] = 0` (the identity for addition), then
 * walk back down the tree.  At each node, we swap the left child with the
 * current value, and add the old left child to the right child.  This
 * distributes the prefix sums correctly.
 *
 * ### Block sums
 * If `block_sums` is non-null, thread 0 saves the total before clearing
 * `temp[kBlockSize-1]`.  This total is needed for the multi-block extension.
 *
 * @param data       In/out array (must have blockDim.x elements per block).
 * @param block_sums If non-null, write the total sum of each block here.
 * @param n          Total number of elements.
 */
__global__ void blelloch_scan_block(float* data, float* block_sums, int n) {
  __shared__ float temp[kBlockSize];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  temp[tid] = (gid < n) ? data[gid] : 0.0F;
  __syncthreads();

  // Up-sweep (reduce)
  for (int stride = 1; stride < kBlockSize; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < kBlockSize) {
      temp[index] += temp[index - stride];
    }
    __syncthreads();
  }

  // Save block sum before clearing last element
  if (tid == 0) {
    if (block_sums != nullptr) {
      block_sums[blockIdx.x] = temp[kBlockSize - 1];
    }
    temp[kBlockSize - 1] = 0.0F;  // exclusive scan: identity element
  }
  __syncthreads();

  // Down-sweep
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < kBlockSize) {
      float t = temp[index - stride];
      temp[index - stride] = temp[index];
      temp[index] += t;
    }
    __syncthreads();
  }

  if (gid < n) data[gid] = temp[tid];
}

// =============================================================================
// Add scanned block sums back to each block
// =============================================================================

__global__ void add_block_sums(float* data, const float* block_sums, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n && blockIdx.x > 0) {
    data[gid] += block_sums[blockIdx.x];
  }
}

// =============================================================================
// Host driver: full exclusive scan
// =============================================================================

void gpu_exclusive_scan(float* d_data, int n) {
  int blocks = (n + kBlockSize - 1) / kBlockSize;

  if (blocks == 1) {
    blelloch_scan_block<<<1, kBlockSize>>>(d_data, nullptr, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }

  // Level 1: scan each block, collect block sums
  float* d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(blocks) * sizeof(float)));

  blelloch_scan_block<<<blocks, kBlockSize>>>(d_data, d_block_sums, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Level 2: scan the block sums themselves
  gpu_exclusive_scan(d_block_sums, blocks);

  // Level 3: add scanned block sums back
  add_block_sums<<<blocks, kBlockSize>>>(d_data, d_block_sums, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_block_sums));
}

int main() {
  constexpr int kN = 1024;
  std::vector<float> h_data(kN);
  for (int i = 0; i < kN; ++i) h_data[static_cast<size_t>(i)] = 1.0F;

  // CPU reference
  std::vector<float> reference(kN);
  std::exclusive_scan(h_data.begin(), h_data.end(), reference.begin(), 0.0F);

  // GPU
  float* d_data = nullptr;
  size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

  gpu_exclusive_scan(d_data, kN);

  std::vector<float> gpu_result(kN);
  CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_data));

  bool ok = true;
  for (int i = 0; i < kN && ok; ++i) {
    if (std::abs(gpu_result[static_cast<size_t>(i)] - reference[static_cast<size_t>(i)]) > 1e-3F) {
      std::fprintf(stderr, "Mismatch at %d: gpu=%.1f cpu=%.1f\n", i,
                   static_cast<double>(gpu_result[static_cast<size_t>(i)]),
                   static_cast<double>(reference[static_cast<size_t>(i)]));
      ok = false;
    }
  }
  std::printf("Exclusive scan: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
