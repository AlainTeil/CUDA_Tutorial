
find_package(CUDAToolkit REQUIRED)

add_executable(32_transformer transformer.cu)
target_link_libraries(32_transformer PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(32_transformer)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(32_transformer_test transformer_test.cu)
  target_link_libraries(32_transformer_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(32_transformer_test)
  gtest_discover_tests(32_transformer_test)
endif()
