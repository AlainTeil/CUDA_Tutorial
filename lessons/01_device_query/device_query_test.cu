/**
 * @file device_query_test.cu
 * @brief Unit tests for Lesson 01 — Device Query.
 *
 * Verifies that the CUDA runtime reports at least one device and that basic
 * device properties are within sane ranges.
 */

#include <gtest/gtest.h>

#include <string>

/// @brief Fixture that queries device 0 properties once.
class DeviceQueryTest : public ::testing::Test {
 protected:
  cudaDeviceProp prop_{};
  int device_count_{0};

  void SetUp() override {
    cudaError_t err = cudaGetDeviceCount(&device_count_);
    ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
    ASSERT_GT(device_count_, 0) << "No CUDA devices available";

    err = cudaGetDeviceProperties(&prop_, 0);
    ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err);
  }
};

TEST_F(DeviceQueryTest, AtLeastOneDevice) {
  EXPECT_GE(device_count_, 1);
}

TEST_F(DeviceQueryTest, DeviceNameNotEmpty) {
  std::string name(prop_.name);
  EXPECT_FALSE(name.empty());
}

TEST_F(DeviceQueryTest, ComputeCapabilityAtLeast75) {
  // Turing (sm_75) is the minimum for CUDA 13.x
  int cc = prop_.major * 10 + prop_.minor;
  EXPECT_GE(cc, 75) << "Compute capability " << prop_.major << "." << prop_.minor
                    << " is below the Turing minimum (7.5)";
}

TEST_F(DeviceQueryTest, HasMultiprocessors) {
  EXPECT_GT(prop_.multiProcessorCount, 0);
}

TEST_F(DeviceQueryTest, WarpSizeIs32) {
  EXPECT_EQ(prop_.warpSize, 32);
}

TEST_F(DeviceQueryTest, MaxThreadsPerBlockAtLeast1024) {
  EXPECT_GE(prop_.maxThreadsPerBlock, 1024);
}

TEST_F(DeviceQueryTest, HasGlobalMemory) {
  EXPECT_GT(prop_.totalGlobalMem, 0UL);
}

TEST_F(DeviceQueryTest, SharedMemoryPerBlockNonZero) {
  EXPECT_GT(prop_.sharedMemPerBlock, 0UL);
}

TEST_F(DeviceQueryTest, SupportsUnifiedAddressing) {
  EXPECT_TRUE(prop_.unifiedAddressing);
}

TEST_F(DeviceQueryTest, SupportsManagedMemory) {
  EXPECT_TRUE(prop_.managedMemory);
}
