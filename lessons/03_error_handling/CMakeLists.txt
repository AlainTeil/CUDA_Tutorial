# Lesson 03 — Error Handling
add_executable(03_error_handling error_handling.cu)
target_link_libraries(03_error_handling PRIVATE CUDA::cudart)
set_project_warnings(03_error_handling)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(03_error_handling_test error_handling_test.cu)
  target_link_libraries(03_error_handling_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(03_error_handling_test)
  gtest_discover_tests(03_error_handling_test)
endif()
