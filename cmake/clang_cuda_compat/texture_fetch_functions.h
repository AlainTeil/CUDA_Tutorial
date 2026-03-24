// Stub header for Clang 18 + CUDA 13+ compatibility.
// CUDA 13 removed texture_fetch_functions.h but Clang 18's
// __clang_cuda_runtime_wrapper.h still references it.
// This empty stub satisfies the #include without side effects.
