# Lesson 02 — Hello Kernel
add_executable(02_hello_kernel hello_kernel.cu)
target_link_libraries(02_hello_kernel PRIVATE CUDA::cudart)
set_project_warnings(02_hello_kernel)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(02_hello_kernel_test hello_kernel_test.cu)
  target_link_libraries(02_hello_kernel_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(02_hello_kernel_test)
  gtest_discover_tests(02_hello_kernel_test)
endif()
