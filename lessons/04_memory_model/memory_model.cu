/**
 * @file memory_model.cu
 * @brief Lesson 04 — CUDA Memory Model: global, constant, and shared memory.
 *
 * Understanding the GPU memory hierarchy is the **single most important**
 * factor in writing fast CUDA code.  This lesson demonstrates the three
 * memory spaces you will use most often.
 *
 * ## Memory hierarchy (from slowest / largest to fastest / smallest)
 *
 * | Space            | Size      | Latency  | Scope          | Lifetime    |
 * |------------------|-----------|----------|----------------|--------------|
 * | Global memory    | GBs       | ~400–600 cycles | All threads | Explicit    |
 * | Constant memory  | 64 KB     | ~4 cycles (cached) | All threads | Explicit |
 * | Shared memory    | 48–96 KB/SM | ~20–30 cycles | Block-local  | Block       |
 * | Registers        | 64K/SM    | 0 cycles | Thread-private | Thread      |
 *
 * ## Part A — Global Memory (vector add)
 *
 * Every `cudaMalloc` allocation lives in **global memory** — the GPU's
 * off-chip DRAM (VRAM).  It has high bandwidth (~400 GB/s on an RTX 3070)
 * but high latency.  Accesses should be **coalesced**: consecutive threads
 * in a warp should read/write consecutive addresses so the hardware can
 * merge them into a single wide transaction.
 *
 * ## Part B — Constant Memory (polynomial evaluation)
 *
 * `__constant__` variables reside in a dedicated 64 KB region that is
 * **cached** and **broadcast** to all threads in a warp in a single cycle
 * (when all threads read the same address).  Ideal for look-up tables,
 * filter coefficients, and hyperparameters that don't change during a kernel.
 *
 * ## Part C — Shared Memory (1-D stencil)
 *
 * `__shared__` memory is a fast, user-managed scratchpad **local to each
 * block**.  Threads within a block use it to share data without going
 * through slow global memory.  You must call `__syncthreads()` after
 * writing to shared memory before any thread reads the written values.
 *
 * ### Halo cells
 * Stencil kernels read neighbours.  At block boundaries, those neighbours
 * belong to a different block’s region.  We load extra "halo" or "ghost"
 * cells into shared memory so every thread can access its neighbours
 * without a second global memory read.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

// =============================================================================
// Macros
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

// =============================================================================
// Part A — Global Memory: vector add
// =============================================================================

/**
 * @brief Element-wise vector addition using global memory.
 *
 * This kernel is **memory-bound**: each element requires 2 loads + 1 store
 * (12 bytes) and only 1 FLOP (addition).  Arithmetic intensity = 1/12 ≈ 0.083
 * FLOP/byte, far below any GPU’s ridge point (see Lesson 22).  Performance
 * is therefore limited by memory bandwidth, not compute.
 *
 * The access pattern is perfectly **coalesced**: thread `i` reads `a[i]` and
 * `b[i]`, and writes `c[i]`.  Consecutive threads in a warp access
 * consecutive 4-byte addresses, which the memory controller merges into a
 * single 128-byte cache-line transaction.
 */
__global__ void vector_add_global(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// =============================================================================
// Part B — Constant Memory: polynomial evaluation
// =============================================================================

/// Polynomial coefficients stored in constant memory (max 64 KB).
///
/// `__constant__` is a CUDA storage qualifier.  The data resides in device
/// DRAM but is cached in a dedicated constant cache.  When all threads in a
/// warp read the **same** address, the cache broadcasts the value to every
/// thread in a single cycle.  If threads read *different* constant addresses
/// in the same cycle, the accesses are serialized — so constant memory is
/// best for **uniform** reads.
__constant__ float kCoeffs[4];

/**
 * @brief Evaluate polynomial  c0 + c1*x + c2*x^2 + c3*x^3  using constant memory.
 *
 * The coefficients `kCoeffs[0..3]` are identical for every thread, so the
 * constant cache gives every warp a broadcast read.  The input `x[idx]`
 * comes from global memory (coalesced).  This is a common pattern in
 * physics simulations and neural networks where the same parameters are
 * applied to many data points.
 */
__global__ void poly_eval_constant(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float xi = x[idx];
    y[idx] = kCoeffs[0] + kCoeffs[1] * xi + kCoeffs[2] * xi * xi + kCoeffs[3] * xi * xi * xi;
  }
}

// =============================================================================
// Part C — Shared Memory: 1-D stencil (3-point average)
// =============================================================================

constexpr int kBlockSize = 256;

/**
 * @brief 1-D stencil computing the average of left, centre, right neighbours.
 *
 * ### Why shared memory?
 * Without shared memory, each thread would read `in[gidx-1]`, `in[gidx]`,
 * and `in[gidx+1]` from global memory — 3 global loads per thread.  But
 * `in[gidx]` for thread `i` is the same as `in[gidx-1]` for thread `i+1`,
 * so most values are read **twice** from slow DRAM.  Loading the tile into
 * shared memory first means each value is read from DRAM exactly once and
 * all neighbour accesses hit the fast on-chip SRAM.
 *
 * ### Halo cells
 * The `+2` in the shared memory declaration (`tile[kBlockSize + 2]`)
 * accounts for the left and right halo elements.  Thread 0 loads the left
 * halo (one element to the left of the block), and the last thread loads
 * the right halo.  Boundary threads that would read outside the array use
 * 0.0 as a sentinel.
 *
 * ### `__syncthreads()`
 * This is a **block-level barrier**.  Every thread in the block must reach
 * this point before any thread can proceed past it.  Without it, some
 * threads might read shared memory locations that haven't been written yet
 * by other threads — a classic **race condition**.
 */
__global__ void stencil_shared(const float* in, float* out, int n) {
  __shared__ float tile[kBlockSize + 2];  // +2 for left and right halo

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int lidx = threadIdx.x + 1;  // offset by 1 for left halo

  // Load centre
  if (gidx < n) {
    tile[lidx] = in[gidx];
  }
  // Load left halo
  if (threadIdx.x == 0) {
    tile[0] = (gidx > 0) ? in[gidx - 1] : 0.0F;
  }
  // Load right halo
  if (threadIdx.x == blockDim.x - 1 || gidx == n - 1) {
    tile[lidx + 1] = (gidx + 1 < n) ? in[gidx + 1] : 0.0F;
  }

  __syncthreads();

  if (gidx < n) {
    out[gidx] = (tile[lidx - 1] + tile[lidx] + tile[lidx + 1]) / 3.0F;
  }
}

// =============================================================================
// main — run all three demos
// =============================================================================
int main() {
  constexpr int kN = 1024;
  constexpr int kThreads = 256;
  int blocks = (kN + kThreads - 1) / kThreads;

  // ---- Part A: Vector Add ---------------------------------------------------
  {
    std::vector<float> ha(kN), hb(kN), hc(kN);
    for (int i = 0; i < kN; ++i) {
      ha[static_cast<size_t>(i)] = static_cast<float>(i);
      hb[static_cast<size_t>(i)] = static_cast<float>(i) * 2.0F;
    }

    float *da, *db, *dc;
    size_t bytes = static_cast<size_t>(kN) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    vector_add_global<<<blocks, kThreads>>>(da, db, dc, kN);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kN && ok; ++i) {
      float expected = static_cast<float>(i) * 3.0F;
      if (std::abs(hc[static_cast<size_t>(i)] - expected) > 1e-5F) ok = false;
    }
    std::printf("Part A (global memory vector add): %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
  }

  // ---- Part B: Constant Memory Polynomial -----------------------------------
  {
    float coeffs[4] = {1.0F, 2.0F, 3.0F, 4.0F};  // 1 + 2x + 3x^2 + 4x^3
    CUDA_CHECK(cudaMemcpyToSymbol(kCoeffs, coeffs, sizeof(coeffs)));

    std::vector<float> hx(kN), hy(kN);
    for (int i = 0; i < kN; ++i) hx[static_cast<size_t>(i)] = static_cast<float>(i) * 0.01F;

    float *dx, *dy;
    size_t bytes = static_cast<size_t>(kN) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    poly_eval_constant<<<blocks, kThreads>>>(dx, dy, kN);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kN && ok; ++i) {
      float xi = hx[static_cast<size_t>(i)];
      float expected = 1.0F + 2.0F * xi + 3.0F * xi * xi + 4.0F * xi * xi * xi;
      if (std::abs(hy[static_cast<size_t>(i)] - expected) > 1e-3F) ok = false;
    }
    std::printf("Part B (constant memory polynomial): %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }

  // ---- Part C: Shared Memory Stencil ----------------------------------------
  {
    std::vector<float> hin(kN), hout(kN);
    for (int i = 0; i < kN; ++i) hin[static_cast<size_t>(i)] = static_cast<float>(i);

    float *din, *dout;
    size_t bytes = static_cast<size_t>(kN) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    stencil_shared<<<blocks, kBlockSize>>>(din, dout, kN);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < kN && ok; ++i) {
      float left = (i > 0) ? hin[static_cast<size_t>(i - 1)] : 0.0F;
      float right = (i < kN - 1) ? hin[static_cast<size_t>(i + 1)] : 0.0F;
      float expected = (left + hin[static_cast<size_t>(i)] + right) / 3.0F;
      if (std::abs(hout[static_cast<size_t>(i)] - expected) > 1e-4F) ok = false;
    }
    std::printf("Part C (shared memory stencil): %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
  }

  return EXIT_SUCCESS;
}
