# Lesson 13 — Activation Functions
add_executable(13_activations activations.cu)
target_link_libraries(13_activations PRIVATE CUDA::cudart)
set_project_warnings(13_activations)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(13_activations_test activations_test.cu)
  target_link_libraries(13_activations_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(13_activations_test)
  gtest_discover_tests(13_activations_test)
endif()
