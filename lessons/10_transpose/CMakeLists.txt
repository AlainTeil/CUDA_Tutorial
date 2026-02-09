# Lesson 10 — Matrix Transpose
add_executable(10_transpose transpose.cu)
target_link_libraries(10_transpose PRIVATE CUDA::cudart)
set_project_warnings(10_transpose)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(10_transpose_test transpose_test.cu)
  target_link_libraries(10_transpose_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(10_transpose_test)
  gtest_discover_tests(10_transpose_test)
endif()
