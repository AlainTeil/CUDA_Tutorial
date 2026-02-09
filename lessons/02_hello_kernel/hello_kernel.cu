/**
 * @file hello_kernel.cu
 * @brief Lesson 02 — Your first CUDA kernel.
 *
 * This is the "Hello, World!" of GPU programming.  We launch a kernel that
 * writes each thread's unique ID into an output array, then copy the result
 * back to the host and verify it.
 *
 * ## The CUDA programming model (key ideas)
 *
 * 1. **Host vs. Device** — The CPU (host) and GPU (device) have separate
 *    memory spaces.  Data must be explicitly copied between them with
 *    `cudaMemcpy` (or use unified memory — see Lesson 06).
 *
 * 2. **`__global__` functions** — These are the *kernels* that run on the
 *    GPU.  They are launched from the host using the `<<<grid, block>>>`
 *    syntax and execute in parallel across thousands of threads.
 *
 * 3. **Thread indexing** — Every thread knows its position:
 *    @code
 *    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *    @endcode
 *    `blockIdx.x`  = which block this thread belongs to (0-based).
 *    `threadIdx.x` = this thread's position within its block.
 *    `blockDim.x`  = number of threads per block.
 *    This formula maps each thread to a unique element in the output.
 *
 * 4. **Bounds checking** — We always launch enough threads to cover `n`
 *    elements, which means some threads in the last block may be "out of
 *    bounds".  The `if (idx < n)` guard prevents out-of-bounds writes.
 *
 * 5. **Memory lifecycle**:
 *    `cudaMalloc` → allocate on device.
 *    `cudaMemcpy` → copy H→D or D→H.
 *    `cudaFree`   → release device memory.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

/**
 * @brief Kernel that writes each thread's global index into the output array.
 *
 * This is the simplest possible kernel: each thread computes its unique
 * 1-D index and stores it.  No thread reads another thread's data, so there
 * are zero data dependencies — this is **embarrassingly parallel**.
 *
 * @param out   Pointer to device memory of size >= total_threads.
 * @param n     Number of elements (used for bounds checking).
 */
__global__ void fill_thread_index(int* out, int n) {
  // Compute the unique global ID for this thread.
  // The formula works for any 1-D grid of 1-D blocks.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

/**
 * @brief Launch the fill_thread_index kernel and copy results back to host.
 *
 * This function encapsulates the full GPU workflow:
 *   1. **cudaMalloc** — reserve device memory.
 *   2. **Compute grid dimensions** — `ceil(n / threads_per_block)` blocks.
 *      The ceiling division `(n + tpb - 1) / tpb` is a standard idiom.
 *   3. **Launch kernel** — the `<<<blocks, threads>>>` syntax is CUDA-specific
 *      and triggers an *asynchronous* dispatch.  The host thread returns
 *      immediately; the GPU processes in the background.
 *   4. **cudaDeviceSynchronize** — block the host until all GPU work finishes.
 *      Without this, the host might read garbage (the copy would start
 *      before the kernel completes).
 *   5. **cudaMemcpy D→H** — pull results back to CPU memory.
 *   6. **cudaFree** — release device memory (just like `free()` on the host).
 *
 * @param n                Number of elements.
 * @param threads_per_block  Threads per block (default: 256).
 * @return  Host vector containing the result.
 */
std::vector<int> launch_fill_thread_index(int n, int threads_per_block = 256) {
  // 1. Allocate device memory.
  int* d_out = nullptr;
  cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int));

  // 2. Compute grid dimensions — enough blocks to cover all n elements.
  int blocks = (n + threads_per_block - 1) / threads_per_block;

  // 3. Launch kernel (asynchronous — host returns immediately).
  fill_thread_index<<<blocks, threads_per_block>>>(d_out, n);

  // 4. Wait for kernel to finish before reading results.
  cudaDeviceSynchronize();

  // 5. Copy result back to host.
  std::vector<int> h_out(static_cast<size_t>(n));
  cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);

  // 6. Free device memory.
  cudaFree(d_out);

  return h_out;
}

int main() {
  constexpr int kN = 1024;

  auto result = launch_fill_thread_index(kN);

  // Verify a few values
  bool ok = true;
  for (int i = 0; i < kN; ++i) {
    if (result[static_cast<size_t>(i)] != i) {
      std::fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n", i, i,
                   result[static_cast<size_t>(i)]);
      ok = false;
      break;
    }
  }

  std::printf("fill_thread_index(%d): %s\n", kN, ok ? "PASSED" : "FAILED");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
