# Lesson 24 — Dropout Regularization
add_executable(24_dropout dropout.cu)
target_link_libraries(24_dropout PRIVATE CUDA::cudart)
set_project_warnings(24_dropout)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(24_dropout_test dropout_test.cu)
  target_link_libraries(24_dropout_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(24_dropout_test)
  gtest_discover_tests(24_dropout_test)
endif()
