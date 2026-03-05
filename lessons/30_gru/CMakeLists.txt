
find_package(CUDAToolkit REQUIRED)

add_executable(30_gru gru.cu)
target_link_libraries(30_gru PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(30_gru)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(30_gru_test gru_test.cu)
  target_link_libraries(30_gru_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(30_gru_test)
  gtest_discover_tests(30_gru_test)
endif()
