/**
 * @file reduction.cu
 * @brief Lesson 08 — Parallel Reduction (sum).
 *
 * Reduction is the fundamental "many → one" parallel primitive: take N
 * inputs and combine them into a single value (sum, max, min, etc.).
 *
 * ## Why it's hard on a GPU
 *
 * Reduction is inherently **sequential** (each step depends on the previous
 * partial result).  The trick is to do it as a balanced **binary tree**:
 *
 * ```
 * Step 0:  [a0 a1 a2 a3 a4 a5 a6 a7]   (8 values)
 * Step 1:  [a0+a1  a2+a3  a4+a5  a6+a7] (4 sums)
 * Step 2:  [a01+a23  a45+a67]            (2 sums)
 * Step 3:  [a0..7]                        (1 sum)
 * ```
 *
 * Each step halves the number of active threads.  For N elements this takes
 * O(log N) steps and O(N) total work — work-efficient.
 *
 * ## Two implementations
 *
 * 1. **Shared-memory tree reduction** — Each block loads into `__shared__`
 *    memory, then threads pair up with "sequential addressing" (stride
 *    halving).  Sequential addressing avoids the bank-conflict issues of
 *    the naive interleaved approach.
 *
 * 2. **Warp-shuffle reduction** — Within a 32-thread warp, threads can
 *    exchange values directly via `__shfl_down_sync` (register-to-register,
 *    no shared memory needed, ~3× faster).  We reduce within each warp
 *    first, store per-warp results to shared memory, then do a final
 *    warp-level reduction on those warp sums.
 *
 * Both produce **per-block partial sums**.  A second (recursive) pass
 * reduces the partial sums to a single scalar.  In production, use CUB or
 * Thrust for highly optimised reductions.
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
// Kernel 1: shared-memory tree reduction (sequential addressing)
// =============================================================================

/**
 * @brief Each block produces one partial sum in `out[blockIdx.x]`.
 *
 * ### Sequential addressing pattern
 * In each step, thread `tid` adds element at `tid + stride` to its own
 * position.  Because `stride` starts at `blockDim.x/2` and halves each
 * iteration, active threads are always contiguous (threads 0..stride-1).
 * This avoids the warp divergence that plagues the "interleaved" pattern
 * (where in the first step only even threads are active, then every 4th, ...).
 */
__global__ void reduce_shared(const float* in, float* out, int n) {
  __shared__ float sdata[kBlockSize];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread loads one element from global memory into shared memory.
  sdata[tid] = (gid < n) ? in[gid] : 0.0F;
  __syncthreads();

  // Tree reduction with sequential addressing.
  // Stride starts at half the block and halves each iteration.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();  // All threads must finish before the next level.
  }

  // Thread 0 writes the block's partial sum.
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// =============================================================================
// Kernel 2: warp-shuffle reduction
// =============================================================================

/**
 * @brief Reduce within a warp using `__shfl_down_sync`.
 *
 * Warp shuffle instructions move data **between registers** of threads in
 * the same warp (32 threads).  No shared memory, no `__syncthreads()`, and
 * only ~1 cycle latency per instruction.
 *
 * `__shfl_down_sync(mask, val, offset)` returns the value of `val` from
 * the thread that is `offset` lanes ahead.  The `mask` (0xFFFFFFFF) means
 * all 32 lanes participate.
 *
 * After 5 iterations (offsets 16, 8, 4, 2, 1), lane 0 holds the sum of
 * all 32 values in the warp.
 */
__device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

/**
 * @brief Block-level reduction: warp shuffles + shared memory for
 *        inter-warp reduction.
 *
 * Strategy:
 *   1. Each warp reduces its 32 values to a single sum (warp shuffle).
 *   2. Lane 0 of each warp writes that sum to shared memory.
 *   3. The first warp loads all warp sums and does one final warp reduce.
 *
 * This approach uses minimal shared memory (blockDim/32 floats) and avoids
 * the log2(blockDim) barriers of the pure shared-memory approach.
 */
__global__ void reduce_warp_shuffle(const float* in, float* out, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (gid < n) ? in[gid] : 0.0F;

  // Intra-warp reduction
  val = warp_reduce_sum(val);

  // First thread in each warp writes to shared memory
  __shared__ float warp_sums[kBlockSize / 32];
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  // First warp reduces the warp sums
  if (warp_id == 0) {
    val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0F;
    val = warp_reduce_sum(val);
  }

  if (threadIdx.x == 0) out[blockIdx.x] = val;
}

// =============================================================================
// Host driver: full reduction
// =============================================================================

/**
 * @brief Sum all elements using the given kernel.
 *
 * Recursively reduces partial sums until a single value remains.
 * Each kernel launch produces `blocks` partial sums.  If `blocks == 1`,
 * we're done; otherwise we recurse.  The recursion depth is
 * O(log_{blockSize}(N)), typically just 2–3 levels.
 *
 * In production, CUB's `DeviceReduce::Sum` does this in a single call
 * with better occupancy tuning and no recursion overhead.
 */
template <typename Kernel>
float gpu_reduce(const float* d_in, int n, Kernel kernel) {
  int blocks = (n + kBlockSize - 1) / kBlockSize;

  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(blocks) * sizeof(float)));

  kernel<<<blocks, kBlockSize>>>(d_in, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  float result;
  if (blocks == 1) {
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    result = gpu_reduce(d_out, blocks, kernel);
  }

  CUDA_CHECK(cudaFree(d_out));
  return result;
}

int main() {
  constexpr int kN = 1 << 20;  // ~1 M
  std::vector<float> h_data(kN);
  for (int i = 0; i < kN; ++i) h_data[static_cast<size_t>(i)] = 1.0F;  // sum = kN

  float* d_data = nullptr;
  size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

  float sum_shared = gpu_reduce(d_data, kN, reduce_shared);
  float sum_warp = gpu_reduce(d_data, kN, reduce_warp_shuffle);
  float sum_cpu = std::accumulate(h_data.begin(), h_data.end(), 0.0F);

  std::printf("CPU sum            : %.0f\n", static_cast<double>(sum_cpu));
  std::printf("GPU (shared mem)   : %.0f\n", static_cast<double>(sum_shared));
  std::printf("GPU (warp shuffle) : %.0f\n", static_cast<double>(sum_warp));

  CUDA_CHECK(cudaFree(d_data));
  return EXIT_SUCCESS;
}
