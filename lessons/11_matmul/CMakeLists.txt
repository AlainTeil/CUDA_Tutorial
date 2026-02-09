# Lesson 11 — Tiled Matrix Multiplication
add_executable(11_matmul matmul.cu)
target_link_libraries(11_matmul PRIVATE CUDA::cudart)
set_project_warnings(11_matmul)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(11_matmul_test matmul_test.cu)
  target_link_libraries(11_matmul_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(11_matmul_test)
  gtest_discover_tests(11_matmul_test)
endif()
