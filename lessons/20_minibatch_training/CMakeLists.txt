cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)

add_executable(20_minibatch_training minibatch_training.cu)
target_link_libraries(20_minibatch_training PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(20_minibatch_training)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(20_minibatch_training_test minibatch_training_test.cu)
  target_link_libraries(20_minibatch_training_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(20_minibatch_training_test)
  gtest_discover_tests(20_minibatch_training_test)
endif()
