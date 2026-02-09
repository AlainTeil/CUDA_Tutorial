# Lesson 12 — Dense (Fully Connected) Layer
add_executable(12_dense_layer dense_layer.cu)
target_link_libraries(12_dense_layer PRIVATE CUDA::cudart)
set_project_warnings(12_dense_layer)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(12_dense_layer_test dense_layer_test.cu)
  target_link_libraries(12_dense_layer_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(12_dense_layer_test)
  gtest_discover_tests(12_dense_layer_test)
endif()
