# Lesson 26 — Cooperative Groups
add_executable(26_cooperative_groups cooperative_groups.cu)
target_link_libraries(26_cooperative_groups PRIVATE CUDA::cudart)
set_project_warnings(26_cooperative_groups)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(26_cooperative_groups_test cooperative_groups_test.cu)
  target_link_libraries(26_cooperative_groups_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(26_cooperative_groups_test)
  gtest_discover_tests(26_cooperative_groups_test)
endif()
