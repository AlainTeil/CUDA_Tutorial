# =============================================================================
# CompilerWarnings.cmake — Apply per-compiler warning flags to a target
# =============================================================================

function(set_project_warnings target_name)
  # ---- C / C++ warnings (GCC & Clang) ----
  set(GCC_CLANG_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wformat=2
  )

  # ---- NVCC host-compiler pass-through ----
  # Note: -Wsign-conversion is omitted for CUDA because kernel launch syntax
  # <<<grid, block>>> inherently converts int to unsigned int, producing
  # unavoidable noise across every kernel call site.
  # Note: -Wpedantic is omitted for CUDA because NVCC's preprocessing pipeline
  # emits GNU-style line markers (# linenum "file") in its intermediate files.
  # Neither GCC nor Clang can selectively silence that pedantic warning,
  # so including -Wpedantic here only creates unavoidable noise.
  set(CUDA_WARNINGS
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-Wshadow
    -Xcompiler=-Wconversion
  )

  # Clang-specific fixups for CUDA:
  #  • -Wconversion implies -Wsign-conversion on Clang (but not GCC), so we
  #    must explicitly disable it to avoid the same unavoidable noise from
  #    kernel launches and dim3 constructors.
  #  • -Wno-gnu-line-marker silences warnings about NVCC's GNU-style line
  #    markers that -Wpedantic (inherited from -Wextra on some Clang versions)
  #    would otherwise flag.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND CUDA_WARNINGS
      -Xcompiler=-Wno-sign-conversion
      -Xcompiler=-Wno-gnu-line-marker
    )
  endif()

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${GCC_CLANG_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${GCC_CLANG_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_WARNINGS}>
  )
endfunction()
