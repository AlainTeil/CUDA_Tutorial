/**
 * @file cublas_cudnn_test.cu
 * @brief Unit tests for Lesson 19 — cuBLAS SGEMM (and cuDNN if available).
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#ifdef HAS_CUDNN
#include <cudnn.h>
#endif

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) FAIL() << "CUDA error: " << cudaGetErrorString(err_); \
  } while (0)

#define CUBLAS_CHECK(call)                                                                 \
  do {                                                                                     \
    cublasStatus_t st_ = (call);                                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) FAIL() << "cuBLAS error: " << static_cast<int>(st_); \
  } while (0)

#ifdef HAS_CUDNN
#define CUDNN_CHECK(call)                                                                   \
  do {                                                                                      \
    cudnnStatus_t st_ = (call);                                                             \
    if (st_ != CUDNN_STATUS_SUCCESS) FAIL() << "cuDNN error: " << cudnnGetErrorString(st_); \
  } while (0)
#endif

// Self-contained function definitions (same as cublas_cudnn.cu)

void cublas_matmul(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M,
                   int K, int N) {
  float alpha = 1.0F, beta = 0.0F;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta,
                           d_C, N));
}

#ifdef HAS_CUDNN
void cudnn_conv2d(cudnnHandle_t cudnn, const float* d_in, const float* d_filter, float* d_out,
                  int C_in, int H, int W, int C_out, int KH, int KW) {
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_in, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out,
                                         C_in, KH, KW));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  int n_out, c_out, h_out, w_out;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n_out, &c_out,
                                                    &h_out, &w_out));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_out,
                                         c_out, h_out, w_out));
  int returned_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf;
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc,
                                                   1, &returned_count, &perf));
  cudnnConvolutionFwdAlgo_t algo = perf.algo;
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc,
                                                      out_desc, algo, &ws_size));
  void* d_ws = nullptr;
  if (ws_size > 0) cudaMalloc(&d_ws, ws_size);
  float alpha = 1.0F, beta = 0.0F;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, in_desc, d_in, filt_desc, d_filter, conv_desc,
                                      algo, d_ws, ws_size, &beta, out_desc, d_out));
  cudaDeviceSynchronize();
  if (d_ws) cudaFree(d_ws);
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}
#endif

// =============================================================================
// cuBLAS SGEMM — identity matrix
// =============================================================================

TEST(CublasTest, IdentityMultiply) {
  constexpr int N = 4;

  // A = identity, B = [1..16]
  std::vector<float> h_A(N * N, 0.0F);
  for (int i = 0; i < N; ++i) h_A[static_cast<size_t>(i) * N + i] = 1.0F;

  std::vector<float> h_B(N * N);
  for (int i = 0; i < N * N; ++i) h_B[static_cast<size_t>(i)] = static_cast<float>(i + 1);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cublas_matmul(handle, d_A, d_B, d_C, N, N, N);

  std::vector<float> h_C(N * N);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  // I * B = B
  for (int i = 0; i < N * N; ++i)
    EXPECT_NEAR(h_C[static_cast<size_t>(i)], h_B[static_cast<size_t>(i)], 1e-5F);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

// =============================================================================
// cuBLAS SGEMM — against CPU reference
// =============================================================================

TEST(CublasTest, AgainstCPU) {
  constexpr int M = 3, K = 4, N = 5;

  std::vector<float> h_A(M * K), h_B(K * N);
  for (int i = 0; i < M * K; ++i) h_A[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.5F;
  for (int i = 0; i < K * N; ++i) h_B[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.3F;

  // CPU reference
  std::vector<float> ref(M * N, 0.0F);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c)
      for (int k = 0; k < K; ++k)
        ref[static_cast<size_t>(r) * N + c] +=
            h_A[static_cast<size_t>(r) * K + k] * h_B[static_cast<size_t>(k) * N + c];

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cublas_matmul(handle, d_A, d_B, d_C, M, K, N);

  std::vector<float> h_C(M * N);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < M * N; ++i)
    EXPECT_NEAR(h_C[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4F) << "i=" << i;

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

// =============================================================================
// cuDNN Conv2D — uniform input + averaging filter = ~1.0
// =============================================================================

#ifdef HAS_CUDNN
TEST(CudnnTest, AveragingFilter) {
  constexpr int C_in = 1, H = 5, W = 5, C_out = 1, KH = 3, KW = 3;
  constexpr int OH = H - KH + 1, OW = W - KW + 1;

  std::vector<float> h_in(C_in * H * W, 1.0F);
  std::vector<float> h_filt(C_out * C_in * KH * KW);
  for (auto& v : h_filt) v = 1.0F / static_cast<float>(KH * KW);

  float *d_in, *d_filt, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_filt, h_filt.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(C_out) * OH * OW * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_filt, h_filt.data(), h_filt.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudnnHandle_t cudnn;
  CUDNN_CHECK(cudnnCreate(&cudnn));
  cudnn_conv2d(cudnn, d_in, d_filt, d_out, C_in, H, W, C_out, KH, KW);

  std::vector<float> h_out(C_out * OH * OW);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < h_out.size(); ++i) EXPECT_NEAR(h_out[i], 1.0F, 1e-5F);

  CUDNN_CHECK(cudnnDestroy(cudnn));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_filt));
  CUDA_CHECK(cudaFree(d_out));
}
#endif
