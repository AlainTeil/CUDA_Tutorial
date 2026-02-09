cmake_minimum_required(VERSION 3.20)

add_executable(17_training training.cu)
target_link_libraries(17_training PRIVATE CUDA::cudart)
set_project_warnings(17_training)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(17_training_test training_test.cu)
  target_link_libraries(17_training_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(17_training_test)
  gtest_discover_tests(17_training_test)
endif()
