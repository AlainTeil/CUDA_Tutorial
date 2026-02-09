# Lesson 08 — Parallel Reduction
add_executable(08_reduction reduction.cu)
target_link_libraries(08_reduction PRIVATE CUDA::cudart)
set_project_warnings(08_reduction)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(08_reduction_test reduction_test.cu)
  target_link_libraries(08_reduction_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(08_reduction_test)
  gtest_discover_tests(08_reduction_test)
endif()
