# Lesson 05 — Thread Hierarchy
add_executable(05_thread_hierarchy thread_hierarchy.cu)
target_link_libraries(05_thread_hierarchy PRIVATE CUDA::cudart)
set_project_warnings(05_thread_hierarchy)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(05_thread_hierarchy_test thread_hierarchy_test.cu)
  target_link_libraries(05_thread_hierarchy_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(05_thread_hierarchy_test)
  gtest_discover_tests(05_thread_hierarchy_test)
endif()
