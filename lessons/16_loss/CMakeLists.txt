cmake_minimum_required(VERSION 3.20)

add_executable(16_loss loss.cu)
target_link_libraries(16_loss PRIVATE CUDA::cudart)
set_project_warnings(16_loss)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(16_loss_test loss_test.cu)
  target_link_libraries(16_loss_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(16_loss_test)
  gtest_discover_tests(16_loss_test)
endif()
