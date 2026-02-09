/**
 * @file cublas_cudnn.cu
 * @brief Lesson 19 — Accelerating DL with cuBLAS and cuDNN.
 *
 * Throughout this tutorial we wrote every kernel by hand.  That is great
 * for learning, but in production you almost always want to call
 * **NVIDIA’s vendor libraries**:
 *
 *   - **cuBLAS**  — dense linear algebra (GEMM, GEMV, etc.).
 *   - **cuDNN**   — deep-learning primitives (convolution, pooling,
 *     batch-norm, activation, RNN, etc.).
 *
 * These libraries are hand-tuned by NVIDIA engineers for each GPU
 * architecture.  They deliver peak performance via:
 *   - Tiled, blocked algorithms that maximise shared-memory reuse.
 *   - Tensor-Core utilisation on Volta+ (FP16/TF32 accumulation).
 *   - Auto-tuning: cuDNN can benchmark multiple algorithms and pick
 *     the fastest for a given problem size.
 *
 * This lesson demonstrates:
 *   1. cuBLAS SGEMM as a drop-in replacement for our hand-written matmul.
 *   2. cuDNN convolution forward (if compiled with HAS_CUDNN).
 *
 * Build with -DCUDA_TUTORIAL_USE_CUDNN=ON to enable the cuDNN section.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef HAS_CUDNN
#include <cudnn.h>
#endif

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                          \
  do {                                                                              \
    cublasStatus_t st_ = (call);                                                    \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                             \
      std::fprintf(stderr, "cuBLAS error at %s:%d — code %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st_));                                          \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

#ifdef HAS_CUDNN
#define CUDNN_CHECK(call)                                                     \
  do {                                                                        \
    cudnnStatus_t st_ = (call);                                               \
    if (st_ != CUDNN_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuDNN error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudnnGetErrorString(st_));                                 \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#endif

// =============================================================================
// Part 1: cuBLAS SGEMM  —  C = alpha * A * B + beta * C
// =============================================================================

/// Compute C[M×N] = A[M×K] * B[K×N] using cublasSgemm.
///
/// **The row-major trick explained:**
///
/// cuBLAS inherited Fortran’s column-major convention, meaning it treats
/// a pointer as a column-major matrix.  Our C/C++ arrays are row-major.
/// Rather than transposing data, we exploit the identity:
///
///   C = A · B  (row-major)
///   is equivalent to
///   C^T = B^T · A^T  (row-major)
///
/// Since a row-major matrix’s memory layout is identical to its
/// column-major transpose, we simply pass B first, then A, and swap
/// M↔N.  cuBLAS sees column-major inputs and produces a column-major
/// output — which, read as row-major, is exactly C.
///
/// Arguments to cublasSgemm(handle, opA, opB, m, n, k, ...):
///   m = N (columns of C when viewed column-major)
///   n = M (rows of C when viewed column-major)
///   k = K (shared dimension)
///   A→B (leading dim N), B→A (leading dim K), C (leading dim N)
void cublas_matmul(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M,
                   int K, int N) {
  float alpha = 1.0F, beta = 0.0F;
  // Row-major trick: call with (N, M, K), B as first arg, A as second.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta,
                           d_C, N));
}

// =============================================================================
// Part 2: cuDNN Convolution (if available)
// =============================================================================

#ifdef HAS_CUDNN

/// Run a 2D convolution:  out = conv(in, filter)
///
/// cuDNN uses a **descriptor-based** API.  Instead of passing raw
/// dimensions as function arguments, you:
///   1. Create descriptor objects (tensor, filter, convolution).
///   2. Set their attributes (shape, data type, padding, stride, etc.).
///   3. Query output dimensions and choose an algorithm.
///   4. Allocate any workspace the algorithm requires.
///   5. Execute.
///   6. Destroy descriptors.
///
/// This is more verbose than a simple function call, but it gives cuDNN
/// the information it needs to pick the best algorithm (direct, FFT,
/// Winograd, implicit GEMM, etc.) and to fuse operations.
///
/// Layout: NCHW (batch, channels, height, width) throughout.
/// in:     [1, C_in, H, W]
/// filter: [C_out, C_in, KH, KW]
/// out:    [1, C_out, OH, OW]
void cudnn_conv2d(cudnnHandle_t cudnn, const float* d_in, const float* d_filter, float* d_out,
                  int C_in, int H, int W, int C_out, int KH, int KW) {
  // Step 1: Create descriptors — cuDNN needs structured metadata, not
  // just raw pointers and ints.  This allows the library to cache plans
  // and reuse them when the same shapes recur (common in training loops).
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // Step 2: Set descriptor attributes.
  // The convolution descriptor specifies padding (0,0), stride (1,1),
  // dilation (1,1), cross-correlation mode, and compute type (FP32).
  // CUDNN_CROSS_CORRELATION matches PyTorch’s nn.Conv2d (rather than
  // true mathematical convolution which flips the kernel).
  CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_in, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out,
                                         C_in, KH, KW));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

  // Step 3: Query the output dimensions — cuDNN computes them from the
  // input, filter, and convolution descriptors (padding, stride, etc.).
  int n_out, c_out, h_out, w_out;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n_out, &c_out,
                                                    &h_out, &w_out));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_out,
                                         c_out, h_out, w_out));

  // Step 4: Choose the best algorithm.  cuDNN benchmarks several
  // implementations (direct, FFT, Winograd, implicit GEMM, …) and
  // returns them ranked by speed.  We take the fastest.
  int returned_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf;
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc,
                                                   1, &returned_count, &perf));
  cudnnConvolutionFwdAlgo_t algo = perf.algo;

  // Step 5: Some algorithms need temporary scratch memory ("workspace").
  // For example, FFT-based convolution needs space to store the
  // frequency-domain representation of the input and filter.
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc,
                                                      out_desc, algo, &ws_size));
  void* d_ws = nullptr;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_size));

  // Step 6: Execute the convolution.  alpha/beta control the formula:
  //   out = alpha * conv(in, filter) + beta * out
  // With alpha=1, beta=0 this is a simple forward convolution.
  float alpha = 1.0F, beta = 0.0F;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, in_desc, d_in, filt_desc, d_filter, conv_desc,
                                      algo, d_ws, ws_size, &beta, out_desc, d_out));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 7: Clean up — free workspace and destroy descriptors.
  if (d_ws) CUDA_CHECK(cudaFree(d_ws));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

#endif  // HAS_CUDNN

// =============================================================================
// Demo — verify library calls produce correct results
// =============================================================================

int main() {
  // --- cuBLAS SGEMM demo ---
  // Multiply a 4×3 matrix by a 3×5 matrix to get a 4×5 result.
  // We fill A and B with simple patterns so the output is easy to verify.
  constexpr int M = 4, K = 3, N = 5;
  std::vector<float> h_A(M * K), h_B(K * N);
  for (int i = 0; i < M * K; ++i) h_A[static_cast<size_t>(i)] = static_cast<float>(i + 1);
  for (int i = 0; i < K * N; ++i) h_B[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.1F;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  // A cuBLAS handle encapsulates library state (stream, math mode, etc.).
  // Create it once, use it for all GEMM calls, destroy it when done.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  cublas_matmul(handle, d_A, d_B, d_C, M, K, N);

  std::vector<float> h_C(M * N);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("cuBLAS SGEMM result (%dx%d):\n", M, N);
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c)
      std::printf("%.3f ", static_cast<double>(h_C[static_cast<size_t>(r) * N + c]));
    std::printf("\n");
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

#ifdef HAS_CUDNN
  // --- cuDNN conv demo ---
  // 1-channel 5×5 input, 1 output channel, 3×3 kernel.
  // The filter is an averaging kernel (each weight = 1/9), so every
  // output pixel should be 1.0 when the input is all-ones.
  constexpr int C_in = 1, H = 5, W = 5, C_out = 1, KH = 3, KW = 3;
  constexpr int OH = H - KH + 1, OW = W - KW + 1;  // valid (no-pad) output

  std::vector<float> h_in(C_in * H * W, 1.0F);
  std::vector<float> h_filt(C_out * C_in * KH * KW);
  for (auto& v : h_filt) v = 1.0F / static_cast<float>(KH * KW);  // averaging kernel

  float *d_in2, *d_filt, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in2, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_filt, h_filt.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(C_out) * OH * OW * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in2, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_filt, h_filt.data(), h_filt.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudnnHandle_t cudnn;
  CUDNN_CHECK(cudnnCreate(&cudnn));

  cudnn_conv2d(cudnn, d_in2, d_filt, d_out, C_in, H, W, C_out, KH, KW);

  std::vector<float> h_out(C_out * OH * OW);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("\ncuDNN Conv2D result (%dx%d):\n", OH, OW);
  for (int r = 0; r < OH; ++r) {
    for (int c = 0; c < OW; ++c)
      std::printf("%.4f ", static_cast<double>(h_out[static_cast<size_t>(r) * OW + c]));
    std::printf("\n");
  }

  CUDNN_CHECK(cudnnDestroy(cudnn));
  CUDA_CHECK(cudaFree(d_in2));
  CUDA_CHECK(cudaFree(d_filt));
  CUDA_CHECK(cudaFree(d_out));
#else
  std::printf("\ncuDNN not available — skipping conv demo.\n");
#endif

  return EXIT_SUCCESS;
}
