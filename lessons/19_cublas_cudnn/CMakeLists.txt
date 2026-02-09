cmake_minimum_required(VERSION 3.20)

# This lesson is only built when CUDA_TUTORIAL_USE_CUDNN is ON
# (checked in the parent CMakeLists.txt before add_subdirectory).

find_package(CUDAToolkit REQUIRED)

add_executable(19_cublas_cudnn cublas_cudnn.cu)
target_link_libraries(19_cublas_cudnn PRIVATE CUDA::cudart CUDA::cublas)
set_project_warnings(19_cublas_cudnn)

# cuDNN is not a CMake CUDA:: target — find it manually.
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDNN_PATH)
find_path(CUDNN_INCLUDE cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV CUDNN_PATH)

if(CUDNN_LIBRARY AND CUDNN_INCLUDE)
  target_link_libraries(19_cublas_cudnn PRIVATE ${CUDNN_LIBRARY})
  target_include_directories(19_cublas_cudnn PRIVATE ${CUDNN_INCLUDE})
  target_compile_definitions(19_cublas_cudnn PRIVATE HAS_CUDNN=1)
else()
  message(WARNING "cuDNN not found — Lesson 19 will build without cuDNN conv demo.")
endif()

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(19_cublas_cudnn_test cublas_cudnn_test.cu)
  target_link_libraries(19_cublas_cudnn_test PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas)
  set_project_warnings(19_cublas_cudnn_test)

  if(CUDNN_LIBRARY AND CUDNN_INCLUDE)
    target_link_libraries(19_cublas_cudnn_test PRIVATE ${CUDNN_LIBRARY})
    target_include_directories(19_cublas_cudnn_test PRIVATE ${CUDNN_INCLUDE})
    target_compile_definitions(19_cublas_cudnn_test PRIVATE HAS_CUDNN=1)
  endif()

  gtest_discover_tests(19_cublas_cudnn_test)
endif()
