/**
 * @file device_query.cu
 * @brief Lesson 01 — Query and display CUDA device properties.
 *
 * Before writing a single kernel it is essential to know **what hardware you
 * are targeting**.  The CUDA runtime exposes a rich `cudaDeviceProp` structure
 * that describes every capability of the GPU: memory sizes, thread limits,
 * warp size, compute capability, and more.
 *
 * ## Why this matters
 *
 * Kernel launch parameters (block size, grid size, shared-memory usage) all
 * depend on hardware limits.  A kernel that uses 96 KB of shared memory will
 * not launch on a device that only offers 48 KB.  Querying the device at
 * startup lets you pick optimal parameters — or fail early with a clear
 * error message instead of a cryptic `cudaErrorLaunchFailure`.
 *
 * ## Key API calls
 *
 * | Function                 | Purpose                             |
 * |--------------------------|-------------------------------------|
 * | `cudaGetDeviceCount`     | How many GPUs are installed?        |
 * | `cudaGetDeviceProperties`| Fill a `cudaDeviceProp` struct      |
 * | `cudaSetDevice`          | Select which GPU to use (Lesson 23+)|
 *
 * ## Selected `cudaDeviceProp` fields
 *
 * - **`totalGlobalMem`** — VRAM size.  This is the hard upper bound on how
 *   much data your kernels can hold on the GPU simultaneously.
 * - **`sharedMemPerBlock`** — Fast on-chip SRAM available per thread block.
 *   Key for tiled algorithms (Lessons 10, 11).
 * - **`warpSize`** — Always 32 on NVIDIA hardware.  The warp is the
 *   fundamental scheduling unit; divergence within a warp is costly.
 * - **`multiProcessorCount`** — Number of Streaming Multiprocessors (SMs).
 *   More SMs = more blocks can execute concurrently.
 * - **`major` / `minor`** — Compute capability.  Determines which
 *   instructions and features (e.g. Tensor Cores, `__half`) are available.
 */

#include <cstdio>
#include <cstdlib>

/// @brief Print properties for a single CUDA device.
/// @param device_id  Ordinal index of the device (0-based).
///
/// We create a stack-allocated `cudaDeviceProp` struct (zero-initialised) and
/// let the runtime fill it.  Each field is a plain C type (int, size_t, char[])
/// so no dynamic allocation is involved — this is a lightweight call.
void print_device_properties(int device_id) {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);

  std::printf("========================================\n");
  std::printf("Device %d: %s\n", device_id, prop.name);
  std::printf("========================================\n");
  std::printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
  std::printf("  Total global memory : %zu MB\n", prop.totalGlobalMem / (1024UL * 1024UL));
  std::printf("  Shared mem / block  : %zu KB\n", prop.sharedMemPerBlock / 1024UL);
  std::printf("  Registers / block   : %d\n", prop.regsPerBlock);
  std::printf("  Warp size           : %d\n", prop.warpSize);
  std::printf("  Max threads / block : %d\n", prop.maxThreadsPerBlock);
  std::printf("  Max block dims      : (%d, %d, %d)\n", prop.maxThreadsDim[0],
              prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  std::printf("  Max grid dims       : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
              prop.maxGridSize[2]);
  std::printf("  Multiprocessors     : %d\n", prop.multiProcessorCount);
  std::printf("  Memory bus width    : %d bits\n", prop.memoryBusWidth);
  std::printf("  L2 cache size       : %d KB\n", prop.l2CacheSize / 1024);
  std::printf("  Concurrent kernels  : %s\n", prop.concurrentKernels ? "yes" : "no");
  std::printf("  Unified addressing  : %s\n", prop.unifiedAddressing ? "yes" : "no");
  std::printf("  Managed memory      : %s\n", prop.managedMemory ? "yes" : "no");
  std::printf("\n");
}

/// @brief Query the number of CUDA devices and return the count.
/// @return Number of CUDA-capable devices, or -1 on error.
///
/// On a system with no NVIDIA driver installed, `cudaGetDeviceCount` returns
/// `cudaErrorNoDevice`.  On a system where the driver version is too old for
/// the CUDA toolkit, it returns `cudaErrorInsufficientDriver`.  In both cases
/// we report the error string and return -1.
int query_device_count() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return count;
}

int main() {
  int count = query_device_count();
  if (count <= 0) {
    std::printf("No CUDA devices found.\n");
    return EXIT_FAILURE;
  }

  std::printf("Found %d CUDA device(s):\n\n", count);
  for (int i = 0; i < count; ++i) {
    print_device_properties(i);
  }
  return EXIT_SUCCESS;
}
