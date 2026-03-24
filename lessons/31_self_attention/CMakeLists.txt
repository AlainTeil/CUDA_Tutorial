
# CUDAToolkit already found at project root.

add_executable(31_self_attention self_attention.cu)
target_link_libraries(31_self_attention PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(31_self_attention)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(31_self_attention_test self_attention_test.cu)
  target_link_libraries(31_self_attention_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(31_self_attention_test)
  gtest_discover_tests(31_self_attention_test TEST_PREFIX "31_self_attention/")
endif()
