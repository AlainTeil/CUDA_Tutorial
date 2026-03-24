
# CUDAToolkit already found at project root.

add_executable(28_embeddings embeddings.cu)
target_link_libraries(28_embeddings PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(28_embeddings)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(28_embeddings_test embeddings_test.cu)
  target_link_libraries(28_embeddings_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(28_embeddings_test)
  gtest_discover_tests(28_embeddings_test)
endif()
