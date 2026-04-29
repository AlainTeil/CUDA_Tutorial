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
# CUDA architecture list  (Turing → Hopper; RTX 3070 = sm_86, H100 = sm_90)
# Override at configure time, e.g. -DCMAKE_CUDA_ARCHITECTURES="86" for a
# faster, single-arch build.  Hopper sm_100 is included for forward-compat.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100")
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
# Default OFF to match every CMakePresets.json preset; enable explicitly with
# -DCUDA_TUTORIAL_BUILD_DOCS=ON if Doxygen is installed.
option(CUDA_TUTORIAL_BUILD_DOCS  "Add 'docs' target for Doxygen" OFF)

# ---------------------------------------------------------------------------
# Host-side UndefinedBehaviorSanitizer (UBSan)
# Catches signed-integer overflow, OOB shifts, misaligned loads, null derefs,
# bad function-pointer casts, etc., in host code.  Device code is unaffected
# (use NVIDIA `compute-sanitizer` for that — see README).
# Enabled via the `ubsan` CMake preset or -DCUDA_TUTORIAL_ENABLE_UBSAN=ON.
# Note: ASan is intentionally not wired up here because it interacts poorly
# with the CUDA runtime initialisation path on many setups.
# ---------------------------------------------------------------------------
option(CUDA_TUTORIAL_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (host code)" OFF)
if(CUDA_TUTORIAL_ENABLE_UBSAN)
  if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "CUDA_TUTORIAL_ENABLE_UBSAN requires GCC or Clang")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=undefined>
    $<$<COMPILE_LANGUAGE:C,CXX>:-fno-sanitize-recover=all>
    $<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=undefined>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-sanitize-recover=all>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>
  )
  add_link_options(-fsanitize=undefined)
  message(STATUS "UBSan enabled for host code (errors abort the program)")
endif()

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
# Minimum 12.0 for C++20 device-code support; the tutorial is regularly
# tested with CUDA 13.x.
# ---------------------------------------------------------------------------
find_package(CUDAToolkit 12.0 REQUIRED)

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
else()
  message(STATUS "clang-format not found; 'format' target disabled")
endif()

# ---------------------------------------------------------------------------
# clang-tidy target  (static analysis; requires compile_commands.json)
# ---------------------------------------------------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(CLANG_TIDY_EXE AND Python3_FOUND)
  file(GLOB_RECURSE TIDY_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/lessons/*.cu
    ${CMAKE_SOURCE_DIR}/lessons/*.cpp
  )
  add_custom_target(tidy
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/cmake/run_clang_tidy.py
            ${CMAKE_BINARY_DIR}
            ${CLANG_TIDY_EXE}
            ${TIDY_SOURCE_FILES}
    COMMENT "Running clang-tidy on all source files (sanitised for CUDA)"
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
