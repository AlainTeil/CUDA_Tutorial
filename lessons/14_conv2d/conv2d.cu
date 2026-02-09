/**
 * @file conv2d.cu
 * @brief Lesson 14 — 2D Convolution: direct & im2col approaches.
 *
 * Convolution is the core operation in Convolutional Neural Networks (CNNs).
 * A small kernel (filter) slides over the input image, computing a dot
 * product at each position to produce one output pixel.
 *
 * ## Direct convolution
 *
 * Each thread computes one output element by iterating over the KH×KW
 * kernel window:
 * @code
 *   out[r][c] = sum_{kr,kc} in[r+kr][c+kc] * kernel[kr][kc]
 * @endcode
 *
 * This is simple but has **non-coalesced** memory accesses (each thread
 * reads a 2-D patch, not consecutive addresses).  It also doesn't leverage
 * the highly optimised GEMM hardware path.
 *
 * ## im2col + GEMM
 *
 * The **im2col** (image to column) transform unfolds every KH×KW input
 * patch into a column of a matrix.  The convolution then becomes a single
 * matrix multiplication:
 *
 * ```
 * col = im2col(in)         # shape (KH*KW) × (OH*OW)
 * out = kernel_row × col   # shape 1 × (OH*OW)  [for 1 output channel]
 * ```
 *
 * For multi-channel / multi-filter convolutions, im2col + GEMM is the
 * standard approach used by cuDNN and most deep learning frameworks because
 * it reuses the highly tuned GEMM kernel.
 *
 * ## Trade-offs
 *
 * | Approach | Pros | Cons |
 * |----------|------|------|
 * | Direct   | Simple, no extra memory | Slower for large kernels |
 * | im2col   | Leverages GEMM, fast  | Uses O(KH·KW·OH·OW) extra memory |
 *
 * This lesson implements single-channel, stride-1, valid-padding convolution.
 * Lesson 19 uses cuDNN for the full multi-channel case.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// =============================================================================
// Direct 2D convolution kernel
// =============================================================================

/// @brief Direct convolution: one thread per output pixel.
///
/// Each thread iterates over the KH×KW kernel window, accumulating a dot
/// product.  The input read pattern is a 2-D sliding window, which means
/// consecutive threads read nearby but not perfectly coalesced addresses.
/// For small kernels (3×3, 5×5) the L1/L2 cache system mitigates this.
///
/// Grid mapping: 2-D — blockIdx.x covers output columns, blockIdx.y covers
/// output rows (same pattern as Lesson 15 pooling).
///
/// @param in      Input image (H × W)
/// @param kernel  Convolution kernel (KH × KW)
/// @param out     Output image (OH × OW), OH = H-KH+1, OW = W-KW+1
/// @param H       Input height
/// @param W       Input width
/// @param KH      Kernel height
/// @param KW      Kernel width
__global__ void conv2d_direct(const float* in, const float* kernel, float* out, int H, int W,
                              int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;

  if (or_ < OH && oc < OW) {
    float sum = 0.0F;
    for (int kr = 0; kr < KH; ++kr) {
      for (int kc = 0; kc < KW; ++kc) {
        sum += in[(or_ + kr) * W + (oc + kc)] * kernel[kr * KW + kc];
      }
    }
    out[or_ * OW + oc] = sum;
  }
}

// =============================================================================
// im2col: unfold input patches into a column matrix
// =============================================================================

/**
 * @brief Unfold patches of size KH×KW into columns (im2col).
 *
 * This is the key transform that converts convolution into GEMM.
 * For each output position (or_, oc), we copy the KH×KW input patch
 * into one column of the output matrix.
 *
 * Layout of `col`:
 *   - Rows: KH*KW  (one row per kernel element)
 *   - Cols: OH*OW  (one column per output position)
 *   - `col[kr*KW+kc][or_*OW+oc] = in[(or_+kr)*W + (oc+kc)]`
 *
 * After im2col, the convolution `out = kernel · col` becomes a GEMM:
 *   kernel has shape (1, KH*KW) for single output channel,
 *   col has shape (KH*KW, OH*OW),
 *   result has shape (1, OH*OW) = the flattened output.
 *
 * Each thread handles one output position (all KH*KW kernel elements).
 */
__global__ void im2col_kernel(const float* in, float* col, int H, int W, int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  int total = OH * OW;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // output position

  if (idx < total) {
    int or_ = idx / OW;
    int oc = idx % OW;
    for (int kr = 0; kr < KH; ++kr) {
      for (int kc = 0; kc < KW; ++kc) {
        col[(kr * KW + kc) * total + idx] = in[(or_ + kr) * W + (oc + kc)];
      }
    }
  }
}

// =============================================================================
// CPU reference
// =============================================================================

void cpu_conv2d(const float* in, const float* kernel, float* out, int H, int W, int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  for (int r = 0; r < OH; ++r)
    for (int c = 0; c < OW; ++c) {
      float sum = 0;
      for (int kr = 0; kr < KH; ++kr)
        for (int kc = 0; kc < KW; ++kc) sum += in[(r + kr) * W + (c + kc)] * kernel[kr * KW + kc];
      out[r * OW + c] = sum;
    }
}

int main() {
  constexpr int H = 8, W = 8, KH = 3, KW = 3;
  constexpr int OH = H - KH + 1, OW = W - KW + 1;

  std::vector<float> h_in(H * W), h_kernel(KH * KW), h_out(OH * OW);
  srand(42);
  for (auto& v : h_in) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));
  for (auto& v : h_kernel) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX));

  // CPU reference
  std::vector<float> h_ref(OH * OW);
  cpu_conv2d(h_in.data(), h_kernel.data(), h_ref.data(), H, W, KH, KW);

  // GPU direct
  float *d_in, *d_kernel, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, H * W * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, KH * KW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, OH * OW * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), H * W * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_kernel, h_kernel.data(), KH * KW * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);
  conv2d_direct<<<blocks, threads>>>(d_in, d_kernel, d_out, H, W, KH, KW);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  float max_err = 0;
  for (int i = 0; i < OH * OW; ++i)
    max_err =
        std::max(max_err, std::abs(h_out[static_cast<size_t>(i)] - h_ref[static_cast<size_t>(i)]));

  std::printf("Conv2D direct  max error: %e\n", static_cast<double>(max_err));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
