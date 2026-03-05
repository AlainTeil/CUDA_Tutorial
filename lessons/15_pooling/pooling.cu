/**
 * @file pooling.cu
 * @brief Lesson 15 — Max Pooling and Average Pooling (forward & backward).
 *
 * Pooling layers serve two roles in a CNN:
 *   1. **Spatial down-sampling** — shrink feature maps so later layers see
 *      wider receptive fields with fewer parameters.
 *   2. **Translation invariance** — small shifts in the input change the
 *      pool output very little (especially max-pool).
 *
 * Two variants are standard:
 *   - Max Pool  — takes the maximum value inside each window.
 *   - Avg Pool  — takes the arithmetic mean of each window.
 *
 * Both operate on a sliding window of size (pool_h × pool_w) that moves
 * across the 2D input with a configurable stride.  The output spatial size
 * is: OH = (H − pool_h) / stride + 1   (similarly for OW).
 *
 * This lesson implements forward *and* backward passes for both variants
 * on a single channel.  The backward passes illustrate a common GPU
 * challenge: gradient **scatter**, where each output element distributes
 * its gradient back to one or more input positions.
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
// Max Pool forward (store indices for backward)
// =============================================================================

/**
 * @brief 2D max-pooling forward pass with argmax index storage.
 *
 * Each thread handles one output position.  It scans the pooling window and
 * records the index of the winning (maximum) element.  Storing that index
 * is critical because the backward pass must know which input produced
 * the max — only that input receives the gradient ("gradient routing").
 *
 * @param in      Input feature map (H × W)
 * @param out     Output feature map (OH × OW)
 * @param indices Output argmax indices for backward pass (OH × OW)
 * @param H       Input height
 * @param W       Input width
 * @param pool_h  Pooling window height
 * @param pool_w  Pooling window width
 * @param stride  Stride of the pooling window
 */
__global__ void maxpool_forward(const float* in, float* out, int* indices, int H, int W, int pool_h,
                                int pool_w, int stride) {
  // Output spatial dimensions — integer arithmetic mirrors the standard
  // pooling formula:  O = (Input − Pool) / Stride + 1.
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;

  int oc = blockIdx.x * blockDim.x + threadIdx.x;   // output column
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;  // output row

  if (or_ < OH && oc < OW) {
    float max_val = -1e30F;
    int max_idx = 0;
    // Slide over the pool_h × pool_w window anchored at (or_*stride, oc*stride).
    for (int ph = 0; ph < pool_h; ++ph) {
      for (int pw = 0; pw < pool_w; ++pw) {
        int ir = or_ * stride + ph;
        int ic = oc * stride + pw;
        float val = in[ir * W + ic];
        if (val > max_val) {
          max_val = val;
          max_idx = ir * W + ic;  // flat index into the input
        }
      }
    }
    int out_idx = or_ * OW + oc;
    out[out_idx] = max_val;
    indices[out_idx] = max_idx;  // save for backward
  }
}

// =============================================================================
// Max Pool backward
// =============================================================================

/**
 * @brief Max-pooling backward: route gradients to max-index positions.
 *
 * The gradient flows only to the element that was selected as the maximum
 * in the forward pass.  All other input positions receive zero gradient.
 * Uses atomicAdd because overlapping windows may target the same input.
 *
 * IMPORTANT: grad_in must be zeroed before calling this kernel.
 *
 * @param grad_out Upstream gradient (OH × OW)
 * @param indices  Argmax indices from the forward pass (OH × OW)
 * @param grad_in  Gradient w.r.t. input (H × W), must be pre-zeroed
 * @param out_size Total number of output elements (OH * OW)
 */
__global__ void maxpool_backward(const float* grad_out, const int* indices, float* grad_in,
                                 int out_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < out_size) {
    // indices[idx] tells us which input element won the max for output idx.
    atomicAdd(&grad_in[indices[idx]], grad_out[idx]);
  }
}

// =============================================================================
// Average Pool forward
// =============================================================================

/**
 * @brief 2D average-pooling forward pass.
 *
 * Computes the arithmetic mean over each pooling window.  Every input
 * element contributes equally, so the backward gradient is distributed
 * uniformly (1 / pool_area per element).
 *
 * @param in      Input feature map (H × W)
 * @param out     Output feature map (OH × OW)
 * @param H       Input height
 * @param W       Input width
 * @param pool_h  Pooling window height
 * @param pool_w  Pooling window width
 * @param stride  Stride of the pooling window
 */
__global__ void avgpool_forward(const float* in, float* out, int H, int W, int pool_h, int pool_w,
                                int stride) {
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;

  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;

  if (or_ < OH && oc < OW) {
    float sum = 0.0F;
    for (int ph = 0; ph < pool_h; ++ph) {
      for (int pw = 0; pw < pool_w; ++pw) {
        sum += in[(or_ * stride + ph) * W + (oc * stride + pw)];
      }
    }
    // Divide by the window area to get the mean.
    out[or_ * OW + oc] = sum / static_cast<float>(pool_h * pool_w);
  }
}

// =============================================================================
// Average Pool backward
// =============================================================================

/**
 * @brief Average-pooling backward: distribute gradient uniformly over pool window.
 *
 * Each input element in the window receives an equal share of the upstream
 * gradient: dL/d(in) += dL/d(out) / (pool_h * pool_w).
 * Uses atomicAdd for correct accumulation with overlapping windows.
 *
 * IMPORTANT: grad_in must be zeroed before calling this kernel.
 *
 * @param grad_out Upstream gradient (OH × OW)
 * @param grad_in  Gradient w.r.t. input (H × W), must be pre-zeroed
 * @param H        Input height
 * @param W        Input width
 * @param pool_h   Pooling window height
 * @param pool_w   Pooling window width
 * @param stride   Stride of the pooling window
 */
__global__ void avgpool_backward(const float* grad_out, float* grad_in, int H, int W, int pool_h,
                                 int pool_w, int stride) {
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;

  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;

  if (or_ < OH && oc < OW) {
    // Each contributing input gets an equal share of the upstream gradient.
    float grad = grad_out[or_ * OW + oc] / static_cast<float>(pool_h * pool_w);
    for (int ph = 0; ph < pool_h; ++ph) {
      for (int pw = 0; pw < pool_w; ++pw) {
        atomicAdd(&grad_in[(or_ * stride + ph) * W + (oc * stride + pw)], grad);
      }
    }
  }
}

int main() {
  // 8×8 input, 2×2 pool, stride 2 → 4×4 output.
  // Non-overlapping windows (stride == pool size) — the most common setting.
  constexpr int H = 8, W = 8, P = 2, S = 2;
  constexpr int OH = (H - P) / S + 1, OW = (W - P) / S + 1;

  std::vector<float> h_in(H * W);
  for (int i = 0; i < H * W; ++i) h_in[static_cast<size_t>(i)] = static_cast<float>(i);

  float* d_in;
  float* d_out;
  int* d_idx;
  CUDA_CHECK(cudaMalloc(&d_in, H * W * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, OH * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_idx, OH * OW * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), H * W * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((OW + 15) / 16, (OH + 15) / 16);

  maxpool_forward<<<blocks, threads>>>(d_in, d_out, d_idx, H, W, P, P, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(OH * OW);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("MaxPool output (%dx%d):\n", OH, OW);
  for (int r = 0; r < OH; ++r) {
    for (int c = 0; c < OW; ++c)
      std::printf("%.0f ", static_cast<double>(h_out[static_cast<size_t>(r) * OW + c]));
    std::printf("\n");
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_idx));
  return EXIT_SUCCESS;
}
