/**
 * @file hello_kernel_test.cu
 * @brief Unit tests for Lesson 02 — Hello Kernel.
 *
 * Verifies that the fill_thread_index kernel correctly writes the global
 * thread index into every element of the output array.
 */

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

// ---------- Kernel (same as lesson source) -----------------------------------

__global__ void fill_thread_index(int* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

// ---------- Helper -----------------------------------------------------------

/// Launch kernel, copy results back.
static std::vector<int> launch(int n, int threads_per_block = 256) {
  int* d_out = nullptr;
  cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(int));
  cudaMemset(d_out, 0, static_cast<size_t>(n) * sizeof(int));

  int blocks = (n + threads_per_block - 1) / threads_per_block;
  fill_thread_index<<<blocks, threads_per_block>>>(d_out, n);
  cudaDeviceSynchronize();

  std::vector<int> h_out(static_cast<size_t>(n));
  cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;
}

// ---------- Tests ------------------------------------------------------------

TEST(HelloKernelTest, SmallArray) {
  constexpr int kN = 32;
  auto result = launch(kN);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(result[static_cast<size_t>(i)], i) << "Mismatch at index " << i;
  }
}

TEST(HelloKernelTest, ExactBlockSize) {
  constexpr int kN = 256;
  auto result = launch(kN);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(result[static_cast<size_t>(i)], i);
  }
}

TEST(HelloKernelTest, NonMultipleOfBlockSize) {
  // 1000 is not a multiple of 256 — tests the bounds-check branch.
  constexpr int kN = 1000;
  auto result = launch(kN);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(result[static_cast<size_t>(i)], i);
  }
}

TEST(HelloKernelTest, LargeArray) {
  constexpr int kN = 1 << 20;  // ~1 M elements
  auto result = launch(kN);

  // Build expected vector
  std::vector<int> expected(static_cast<size_t>(kN));
  std::iota(expected.begin(), expected.end(), 0);

  EXPECT_EQ(result, expected);
}

TEST(HelloKernelTest, SingleElement) {
  auto result = launch(1);
  ASSERT_EQ(result.size(), 1UL);
  EXPECT_EQ(result[0], 0);
}
