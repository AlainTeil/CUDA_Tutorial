# Lesson 14 — Convolution 2D
add_executable(14_conv2d conv2d.cu)
target_link_libraries(14_conv2d PRIVATE CUDA::cudart)
set_project_warnings(14_conv2d)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(14_conv2d_test conv2d_test.cu)
  target_link_libraries(14_conv2d_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(14_conv2d_test)
  gtest_discover_tests(14_conv2d_test)
endif()
