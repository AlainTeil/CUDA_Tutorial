# Lesson 23 — Batch Normalization
add_executable(23_batch_normalization batch_normalization.cu)
target_link_libraries(23_batch_normalization PRIVATE CUDA::cudart)
set_project_warnings(23_batch_normalization)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(23_batch_normalization_test batch_normalization_test.cu)
  target_link_libraries(23_batch_normalization_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(23_batch_normalization_test)
  gtest_discover_tests(23_batch_normalization_test TEST_PREFIX "23_batch_normalization/")
endif()
