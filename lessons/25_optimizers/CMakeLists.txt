# Lesson 25 — Optimizers: Adam, AdamW & Learning-Rate Schedulers
add_executable(25_optimizers optimizers.cu)
target_link_libraries(25_optimizers PRIVATE CUDA::cudart)
set_project_warnings(25_optimizers)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(25_optimizers_test optimizers_test.cu)
  target_link_libraries(25_optimizers_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(25_optimizers_test)
  gtest_discover_tests(25_optimizers_test TEST_PREFIX "25_optimizers/")
endif()
