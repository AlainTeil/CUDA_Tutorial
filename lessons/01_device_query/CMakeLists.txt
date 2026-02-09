# Lesson 01 — Device Query
add_executable(01_device_query device_query.cu)
target_link_libraries(01_device_query PRIVATE CUDA::cudart)
set_project_warnings(01_device_query)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(01_device_query_test device_query_test.cu)
  target_link_libraries(01_device_query_test PRIVATE GTest::gtest_main GTest::gmock CUDA::cudart)
  set_project_warnings(01_device_query_test)
  gtest_discover_tests(01_device_query_test)
endif()
