/**
 * @file error_handling.cu
 * @brief Lesson 03 — Robust CUDA error handling.
 *
 * CUDA API calls report errors by returning `cudaError_t` (an enum).  If you
 * ignore these return codes, bugs become **silent** — kernels produce wrong
 * results, memory is never allocated, copies go nowhere — and you only notice
 * many lines later (or never).
 *
 * ## Two kinds of errors
 *
 * 1. **Synchronous** — returned directly by runtime calls like `cudaMalloc`,
 *    `cudaMemcpy`, `cudaStreamCreate`.  Check immediately.
 *
 * 2. **Asynchronous** — kernel launches (`<<<>>>`) return immediately on the
 *    host.  If the launch configuration is invalid (too many threads, too
 *    much shared memory), the error is stored and can be retrieved with:
 *    @code
 *    cudaGetLastError();          // returns *and clears* the error
 *    cudaPeekAtLastError();       // returns but does NOT clear
 *    cudaDeviceSynchronize();     // also surfaces kernel-execution errors
 *    @endcode
 *
 * ## Macro vs. exception
 *
 * This lesson defines two styles:
 * - `CUDA_CHECK(call)` — prints file/line and calls `std::abort()`.  Simple,
 *   good for tutorials and scripts where there is no recovery path.
 * - `CUDA_CHECK_THROW(call)` — throws `std::runtime_error`.  Better for
 *   library code where the caller may want to catch and recover.
 *
 * Every subsequent lesson in this tutorial uses `CUDA_CHECK`.
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// =============================================================================
// CUDA_CHECK — abort on error (suitable for tutorials / quick scripts)
// =============================================================================

/**
 * @def CUDA_CHECK(call)
 * @brief Wrap a CUDA runtime API call; print file/line on failure and abort.
 *
 * The `do { ... } while (0)` wrapper is a standard C/C++ idiom that makes the
 * macro safe to use in any statement context (e.g. inside `if` without braces).
 * `__FILE__` and `__LINE__` are expanded by the preprocessor at the call site,
 * so the error message points to the exact source location.
 */
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

/**
 * @def CUDA_CHECK_LAST()
 * @brief Check for the last asynchronous CUDA error (e.g. after a kernel launch).
 *
 * Use this immediately after a `<<<>>>` launch to catch configuration errors
 * (wrong block size, too much shared memory, etc.).  Note that execution
 * errors inside the kernel itself are only surfaced after a synchronisation
 * point (`cudaDeviceSynchronize`, `cudaStreamSynchronize`, or event sync).
 */
#define CUDA_CHECK_LAST()                                                           \
  do {                                                                              \
    cudaError_t err_ = cudaGetLastError();                                          \
    if (err_ != cudaSuccess) {                                                      \
      std::fprintf(stderr, "CUDA kernel error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                       \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

// =============================================================================
// Exception-based variant
// =============================================================================

/**
 * @brief Throw std::runtime_error on CUDA failure.
 *
 * Suitable for library code where callers may want to catch and recover.
 *
 * @param err   cudaError_t returned by the API call.
 * @param file  Source file name (__FILE__).
 * @param line  Source line number (__LINE__).
 */
inline void cuda_check_throw(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::string msg = std::string("CUDA error at ") + file + ":" + std::to_string(line) + " — " +
                      cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }
}

/// Exception-based variant of CUDA_CHECK.
#define CUDA_CHECK_THROW(call) cuda_check_throw((call), __FILE__, __LINE__)

// =============================================================================
// Demo kernels
// =============================================================================

/// Trivial kernel — does nothing useful.
__global__ void dummy_kernel() {}

/// Kernel we will intentionally misconfigure to provoke an error.
__global__ void write_value(int* out) {
  out[0] = 42;
}

// =============================================================================
// main
// =============================================================================
int main() {
  // 1. Normal usage — query device
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::printf("Device count: %d\n", count);

  // 2. Launch a valid kernel
  dummy_kernel<<<1, 1>>>();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("dummy_kernel launched successfully.\n");

  // 3. Demonstrate catching an error (allocate zero bytes is ok, but let's show
  //    error path with an intentionally bad cudaMemcpy direction)
  int* d_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));

  // Exception-based: demonstrate try/catch
  try {
    // This call intentionally uses a wrong direction to trigger an error.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    CUDA_CHECK_THROW(cudaMemcpy(d_ptr, d_ptr, sizeof(int), static_cast<cudaMemcpyKind>(999)));
#pragma GCC diagnostic pop
  } catch (const std::runtime_error& e) {
    std::printf("Caught expected exception: %s\n", e.what());
  }

  CUDA_CHECK(cudaFree(d_ptr));
  std::printf("Error handling demo complete.\n");
  return EXIT_SUCCESS;
}
