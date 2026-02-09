# Lesson 22 — Performance Measurement & Benchmarking
add_executable(22_performance_measurement performance_measurement.cu)
target_link_libraries(22_performance_measurement PRIVATE CUDA::cudart)
set_project_warnings(22_performance_measurement)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(22_performance_measurement_test performance_measurement_test.cu)
  target_link_libraries(22_performance_measurement_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(22_performance_measurement_test)
  gtest_discover_tests(22_performance_measurement_test)
endif()
