cmake_minimum_required(VERSION 3.20)

add_executable(18_inference inference.cu)
target_link_libraries(18_inference PRIVATE CUDA::cudart)
set_project_warnings(18_inference)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(18_inference_test inference_test.cu)
  target_link_libraries(18_inference_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(18_inference_test)
  gtest_discover_tests(18_inference_test)
endif()
