/**
 * @file error_handling_test.cu
 * @brief Unit tests for Lesson 03 — Error Handling.
 *
 * Verifies that:
 *  - CUDA_CHECK_THROW throws on bad API calls.
 *  - CUDA_CHECK_THROW does NOT throw on valid API calls.
 *  - cudaGetLastError correctly surfaces kernel launch errors.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

// ---------- Reuse the helpers from the lesson source -------------------------

inline void cuda_check_throw(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::string msg = std::string("CUDA error at ") + file + ":" + std::to_string(line) + " — " +
                      cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }
}

#define CUDA_CHECK_THROW(call) cuda_check_throw((call), __FILE__, __LINE__)

// ---------- Kernels ----------------------------------------------------------

__global__ void trivial_kernel(int* out) {
  out[0] = 42;
}

// ---------- Tests ------------------------------------------------------------

TEST(ErrorHandlingTest, ValidMallocNoThrow) {
  int* d_ptr = nullptr;
  EXPECT_NO_THROW(CUDA_CHECK_THROW(cudaMalloc(&d_ptr, sizeof(int))));
  EXPECT_NO_THROW(CUDA_CHECK_THROW(cudaFree(d_ptr)));
}

TEST(ErrorHandlingTest, InvalidMemcpyKindThrows) {
  int* d_ptr = nullptr;
  CUDA_CHECK_THROW(cudaMalloc(&d_ptr, sizeof(int)));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
  EXPECT_THROW(
      CUDA_CHECK_THROW(cudaMemcpy(d_ptr, d_ptr, sizeof(int), static_cast<cudaMemcpyKind>(999))),
      std::runtime_error);
#pragma GCC diagnostic pop

  // Reset the error state so subsequent tests are not affected
  cudaGetLastError();
  CUDA_CHECK_THROW(cudaFree(d_ptr));
}

TEST(ErrorHandlingTest, KernelLaunchSucceeds) {
  int* d_ptr = nullptr;
  CUDA_CHECK_THROW(cudaMalloc(&d_ptr, sizeof(int)));

  trivial_kernel<<<1, 1>>>(d_ptr);
  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess);

  CUDA_CHECK_THROW(cudaDeviceSynchronize());

  int h_val = 0;
  CUDA_CHECK_THROW(cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_val, 42);

  CUDA_CHECK_THROW(cudaFree(d_ptr));
}

TEST(ErrorHandlingTest, InvalidLaunchConfigDetected) {
  int* d_ptr = nullptr;
  CUDA_CHECK_THROW(cudaMalloc(&d_ptr, sizeof(int)));

  // Launch with 0 threads — invalid configuration
  trivial_kernel<<<1, 0>>>(d_ptr);
  cudaError_t err = cudaGetLastError();
  EXPECT_NE(err, cudaSuccess);

  // Reset error state
  cudaGetLastError();
  CUDA_CHECK_THROW(cudaFree(d_ptr));
}

TEST(ErrorHandlingTest, ExceptionMessageContainsFileAndLine) {
  try {
    CUDA_CHECK_THROW(cudaErrorInvalidValue);
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    // Should contain file name
    EXPECT_NE(msg.find("error_handling_test.cu"), std::string::npos);
    // Should contain the cuda error description
    EXPECT_NE(msg.find("invalid argument"), std::string::npos);
  }
}
