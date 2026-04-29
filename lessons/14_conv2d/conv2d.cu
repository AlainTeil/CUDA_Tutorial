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
    const cudaError_t err_ = (call);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// =============================================================================
// Direct 2D convolution kernel
// =============================================================================

/**
 * @brief Direct convolution: one thread per output pixel.
 *
 * Each thread iterates over the KH×KW kernel window, accumulating a dot
 * product.  The input read pattern is a 2-D sliding window, which means
 * consecutive threads read nearby but not perfectly coalesced addresses.
 * For small kernels (3×3, 5×5) the L1/L2 cache system mitigates this.
 *
 * Grid mapping: 2-D — blockIdx.x covers output columns, blockIdx.y covers
 * output rows (same pattern as Lesson 15 pooling).
 *
 * @param in      Input image (H × W)
 * @param kernel  Convolution kernel (KH × KW)
 * @param out     Output image (OH × OW), OH = H-KH+1, OW = W-KW+1
 * @param H       Input height
 * @param W       Input width
 * @param KH      Kernel height
 * @param KW      Kernel width
 */
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
 *
 * @param in   Input image (H × W)
 * @param col  Output column matrix (KH*KW × OH*OW)
 * @param H    Input height
 * @param W    Input width
 * @param KH   Kernel height
 * @param KW   Kernel width
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
// Tiled GEMM (single-precision, row-major)
// =============================================================================
//
// Self-contained copy of the tiled matrix multiply from Lesson 11, used here
// to complete the im2col → GEMM convolution pipeline.  Each block computes a
// kTile×kTile tile of C; threads cooperatively load A and B tiles into
// shared memory and accumulate a partial dot product across K.
//
//   C[M×N] = A[M×K] · B[K×N]
//
// In the convolution context (one output channel):
//   A = kernel (1 × KH*KW)
//   B = im2col (KH*KW × OH*OW)
//   C = output (1 × OH*OW)
//
// Pedagogical note: a production implementation would call cuBLAS sgemm
// (Lesson 19) instead of rolling its own GEMM.

constexpr int kTile = 16;

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0F;

  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    int a_col = t * kTile + threadIdx.x;
    int b_row = t * kTile + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0F;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0F;
    __syncthreads();

    for (int k = 0; k < kTile; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }

  if (row < M && col < N) C[row * N + col] = acc;
}

/**
 * @brief End-to-end convolution via im2col + tiled GEMM.
 *
 * Allocates a temporary [(KH*KW) × (OH*OW)] column buffer, runs im2col to
 * fill it, then computes `out = kernel · col` using the tiled matmul above.
 * The `col` buffer is the storage cost the table in the file header warns
 * about: O(KH · KW · OH · OW) extra memory in exchange for a hardware-
 * friendly GEMM.
 *
 * @param d_in     Input image on device (H × W).
 * @param d_kernel Convolution kernel on device (KH × KW).
 * @param d_out    Output image on device (OH × OW), pre-allocated.
 * @param H,W      Input dimensions.
 * @param KH,KW    Kernel dimensions.
 */
void conv2d_im2col_gemm(const float* d_in, const float* d_kernel, float* d_out, int H, int W,
                        int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  const int K = KH * KW;
  const int N = OH * OW;

  float* d_col = nullptr;
  CUDA_CHECK(cudaMalloc(&d_col, static_cast<size_t>(K) * N * sizeof(float)));

  constexpr int kIm2colBlock = 256;
  const int im2col_grid = (N + kIm2colBlock - 1) / kIm2colBlock;
  im2col_kernel<<<im2col_grid, kIm2colBlock>>>(d_in, d_col, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());

  // GEMM: out[1, N] = kernel[1, K] * col[K, N]
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (1 + kTile - 1) / kTile);
  matmul_tiled<<<blocks, threads>>>(d_kernel, d_col, d_out, /*M=*/1, N, K);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_col));
}

// =============================================================================
// Backward pass — gradients w.r.t. kernel and input
// =============================================================================
//
// Forward (valid, stride 1, single channel):
//
//     out[r][c] = sum_{kr,kc} in[r+kr][c+kc] * kernel[kr][kc]
//
// Differentiating w.r.t. each parameter gives:
//
//     dkernel[kr][kc] = sum_{r,c}    dout[r][c] * in[r+kr][c+kc]
//     din   [i ][j ]  = sum_{kr,kc}  dout[i-kr][j-kc] * kernel[kr][kc]
//
// `dkernel` is a cross-correlation of `dout` with `in` (small output, one
// thread per kernel weight).  `din` is the *full* convolution of `dout`
// with the spatially-rotated kernel; we compute it by accumulating
// `dout[r][c] * kernel[kr][kc]` into `din[r+kr][c+kc]`, which avoids
// boundary case-analysis since every (r,c,kr,kc) maps to a valid `din`
// index by construction.
//
// One block per kernel weight for `dkernel`; one thread per `dout` element
// for `din` (each thread scatters into the KH*KW input pixels it touched
// on the forward pass via atomic adds).

/**
 * @brief Gradient of the loss w.r.t. each kernel weight.
 *
 * Grid mapping: one thread per kernel weight (KH*KW total).  Each thread
 * sweeps the full OH x OW output gradient and accumulates the dot product
 * with the matching shifted input window.
 *
 * @param in        Forward input (H x W).
 * @param dout      Upstream gradient (OH x OW).
 * @param dkernel   Output gradient w.r.t. kernel (KH x KW).
 * @param H,W       Input dimensions.
 * @param KH,KW     Kernel dimensions.
 */
__global__ void conv2d_backward_dkernel(const float* __restrict__ in,
                                        const float* __restrict__ dout, float* __restrict__ dkernel,
                                        int H, int W, int KH, int KW) {
  int kc = blockIdx.x * blockDim.x + threadIdx.x;
  int kr = blockIdx.y * blockDim.y + threadIdx.y;
  if (kr >= KH || kc >= KW) return;

  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  float acc = 0.0F;
  for (int r = 0; r < OH; ++r) {
    for (int c = 0; c < OW; ++c) {
      acc += dout[r * OW + c] * in[(r + kr) * W + (c + kc)];
    }
  }
  dkernel[kr * KW + kc] = acc;
}

/**
 * @brief Gradient of the loss w.r.t. each input pixel.
 *
 * Implemented as a scatter from the output-gradient grid: each (r,c) in
 * `dout` contributes `dout[r][c] * kernel[kr][kc]` to `din[r+kr][c+kc]`
 * for every (kr,kc) in the kernel.  Atomic adds resolve the writes from
 * threads whose receptive fields overlap.
 *
 * @param dout      Upstream gradient (OH x OW).
 * @param kernel    Forward kernel (KH x KW).
 * @param din       Output gradient w.r.t. input (H x W), pre-zeroed by caller.
 * @param H,W       Input dimensions.
 * @param KH,KW     Kernel dimensions.
 */
__global__ void conv2d_backward_dx(const float* __restrict__ dout, const float* __restrict__ kernel,
                                   float* din, int H, int W, int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ >= OH || oc >= OW) return;

  const float g = dout[or_ * OW + oc];
  for (int kr = 0; kr < KH; ++kr) {
    for (int kc = 0; kc < KW; ++kc) {
      atomicAdd(&din[(or_ + kr) * W + (oc + kc)], g * kernel[kr * KW + kc]);
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
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  float max_err = 0;
  for (int i = 0; i < OH * OW; ++i)
    max_err =
        std::max(max_err, std::abs(h_out[static_cast<size_t>(i)] - h_ref[static_cast<size_t>(i)]));

  std::printf("Conv2D direct  max error: %e\n", static_cast<double>(max_err));

  // ----- im2col + GEMM path -----
  std::vector<float> h_out_gemm(OH * OW);
  conv2d_im2col_gemm(d_in, d_kernel, d_out, H, W, KH, KW);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out_gemm.data(), d_out, OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  float max_err_gemm = 0;
  for (int i = 0; i < OH * OW; ++i)
    max_err_gemm = std::max(
        max_err_gemm, std::abs(h_out_gemm[static_cast<size_t>(i)] - h_ref[static_cast<size_t>(i)]));
  std::printf("Conv2D im2col+GEMM max error: %e\n", static_cast<double>(max_err_gemm));

  // ----- Backward demo (synthetic upstream gradient = 1) -----
  float *d_dout, *d_dkernel, *d_dx;
  CUDA_CHECK(cudaMalloc(&d_dout, OH * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dkernel, KH * KW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, H * W * sizeof(float)));
  std::vector<float> h_dout(OH * OW, 1.0F);
  CUDA_CHECK(cudaMemcpy(d_dout, h_dout.data(), OH * OW * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dx, 0, H * W * sizeof(float)));

  dim3 k_threads(KW, KH);
  dim3 k_blocks(1, 1);
  conv2d_backward_dkernel<<<k_blocks, k_threads>>>(d_in, d_dout, d_dkernel, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  conv2d_backward_dx<<<blocks, threads>>>(d_dout, d_kernel, d_dx, H, W, KH, KW);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dkernel(KH * KW);
  CUDA_CHECK(
      cudaMemcpy(h_dkernel.data(), d_dkernel, KH * KW * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Conv2D backward dkernel[0..2] = %.4f, %.4f, %.4f\n",
              static_cast<double>(h_dkernel[0]), static_cast<double>(h_dkernel[1]),
              static_cast<double>(h_dkernel[2]));

  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_dkernel));
  CUDA_CHECK(cudaFree(d_dx));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
