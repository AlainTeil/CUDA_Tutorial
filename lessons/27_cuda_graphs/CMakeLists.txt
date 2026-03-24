# Lesson 27 — CUDA Graphs
add_executable(27_cuda_graphs cuda_graphs.cu)
target_link_libraries(27_cuda_graphs PRIVATE CUDA::cudart)
set_project_warnings(27_cuda_graphs)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(27_cuda_graphs_test cuda_graphs_test.cu)
  target_link_libraries(27_cuda_graphs_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(27_cuda_graphs_test)
  gtest_discover_tests(27_cuda_graphs_test TEST_PREFIX "27_cuda_graphs/")
endif()
