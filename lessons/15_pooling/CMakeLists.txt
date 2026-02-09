# Lesson 15 — Pooling Layers
add_executable(15_pooling pooling.cu)
target_link_libraries(15_pooling PRIVATE CUDA::cudart)
set_project_warnings(15_pooling)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(15_pooling_test pooling_test.cu)
  target_link_libraries(15_pooling_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(15_pooling_test)
  gtest_discover_tests(15_pooling_test)
endif()
