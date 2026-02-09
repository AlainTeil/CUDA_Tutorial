# =============================================================================
# CUDA Tutorial — Root CMakeLists.txt
# =============================================================================
cmake_minimum_required(VERSION 3.20)

project(CUDA_Tutorial
  VERSION   1.0.0
  LANGUAGES C CXX CUDA
  DESCRIPTION "A comprehensive CUDA programming tutorial for deep learning"
)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD          17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

set(CMAKE_CUDA_STANDARD          20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# CUDA architecture list  (Turing → Hopper; RTX 3070 = sm_86)
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# ---------------------------------------------------------------------------
# Build-type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ---------------------------------------------------------------------------
# Project-wide compiler warnings
# ---------------------------------------------------------------------------
include(cmake/CompilerWarnings.cmake)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CUDA_TUTORIAL_USE_CUDNN "Enable cuDNN/cuBLAS lesson (Lesson 19)" OFF)
option(CUDA_TUTORIAL_BUILD_TESTS "Build unit tests" ON)
option(CUDA_TUTORIAL_BUILD_DOCS  "Add 'docs' target for Doxygen" ON)

# ---------------------------------------------------------------------------
# GoogleTest via FetchContent
# ---------------------------------------------------------------------------
if(CUDA_TUTORIAL_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  # Prevent GoogleTest from overriding compiler/linker settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)
endif()

# ---------------------------------------------------------------------------
# CUDA Toolkit imported targets (cublas, cudnn, cudart, …)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------
add_subdirectory(lessons)

# ---------------------------------------------------------------------------
# clang-format target
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/lessons/*.cu
    ${CMAKE_SOURCE_DIR}/lessons/*.cpp
    ${CMAKE_SOURCE_DIR}/lessons/*.h
    ${CMAKE_SOURCE_DIR}/lessons/*.cuh
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
    VERBATIM
  )
endif()

# ---------------------------------------------------------------------------
# Doxygen target
# ---------------------------------------------------------------------------
if(CUDA_TUTORIAL_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  else()
    message(STATUS "Doxygen not found — 'docs' target will not be available")
  endif()
endif()
