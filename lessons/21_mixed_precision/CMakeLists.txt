cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)

add_executable(21_mixed_precision mixed_precision.cu)
target_link_libraries(21_mixed_precision PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(21_mixed_precision)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(21_mixed_precision_test mixed_precision_test.cu)
  target_link_libraries(21_mixed_precision_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(21_mixed_precision_test)
  gtest_discover_tests(21_mixed_precision_test)
endif()
