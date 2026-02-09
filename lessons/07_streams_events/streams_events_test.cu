/**
 * @file streams_events_test.cu
 * @brief Unit tests for Lesson 07 — Streams & Events.
 *
 * Verifies that multi-stream execution produces the same results as serial.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err_ = (call);                                \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

__global__ void scale_kernel(float* data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] *= factor;
}

// Helper: run with num_streams
static std::vector<float> run_streams(int n, float factor, int num_streams) {
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float* h_data = nullptr;
  cudaMallocHost(&h_data, bytes);
  for (int i = 0; i < n; ++i) h_data[i] = static_cast<float>(i);

  float* d_data = nullptr;
  cudaMalloc(&d_data, bytes);

  int chunk_size = (n + num_streams - 1) / num_streams;
  std::vector<cudaStream_t> streams(static_cast<size_t>(num_streams));
  for (auto& s : streams) cudaStreamCreate(&s);

  for (int i = 0; i < num_streams; ++i) {
    int offset = i * chunk_size;
    int size = (offset + chunk_size <= n) ? chunk_size : (n - offset);
    if (size <= 0) break;
    size_t cb = static_cast<size_t>(size) * sizeof(float);
    cudaMemcpyAsync(d_data + offset, h_data + offset, cb, cudaMemcpyHostToDevice,
                    streams[static_cast<size_t>(i)]);
    scale_kernel<<<(size + 255) / 256, 256, 0, streams[static_cast<size_t>(i)]>>>(d_data + offset,
                                                                                  factor, size);
    cudaMemcpyAsync(h_data + offset, d_data + offset, cb, cudaMemcpyDeviceToHost,
                    streams[static_cast<size_t>(i)]);
  }
  cudaDeviceSynchronize();

  std::vector<float> result(h_data, h_data + n);

  for (auto& s : streams) cudaStreamDestroy(s);
  cudaFree(d_data);
  cudaFreeHost(h_data);
  return result;
}

// ---- Tests ------------------------------------------------------------------

class StreamsTest : public ::testing::TestWithParam<int> {};

TEST_P(StreamsTest, ResultMatchesSerial) {
  int n = GetParam();
  constexpr float kFactor = 2.5F;

  // Serial = 1 stream
  auto serial = run_streams(n, kFactor, 1);
  // Concurrent = 4 streams
  auto concurrent = run_streams(n, kFactor, 4);

  ASSERT_EQ(serial.size(), concurrent.size());
  for (size_t i = 0; i < serial.size(); ++i) {
    EXPECT_NEAR(serial[i], concurrent[i], 1e-5F) << "at index " << i;
  }
}

TEST_P(StreamsTest, CorrectValues) {
  int n = GetParam();
  constexpr float kFactor = 3.0F;
  auto result = run_streams(n, kFactor, 4);

  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i) * kFactor;
    EXPECT_NEAR(result[static_cast<size_t>(i)], expected, 1e-5F) << "at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, StreamsTest, ::testing::Values(1, 100, 1000, 65537));

TEST(StreamsEventTest, EventTimingNonNegative) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  EXPECT_GE(ms, 0.0F);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}
