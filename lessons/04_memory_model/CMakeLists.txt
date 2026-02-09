# Lesson 04 — Memory Model
add_executable(04_memory_model memory_model.cu)
target_link_libraries(04_memory_model PRIVATE CUDA::cudart)
set_project_warnings(04_memory_model)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(04_memory_model_test memory_model_test.cu)
  target_link_libraries(04_memory_model_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(04_memory_model_test)
  gtest_discover_tests(04_memory_model_test)
endif()
