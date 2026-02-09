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

/// Each thread handles one output position.  It scans the pooling window and
/// records the *index* of the winning (maximum) element.  Storing that index
/// is critical because the backward pass must know **which** input produced
/// the max — only that input receives the gradient ("gradient routing").
///
/// Grid mapping: 2D — blockIdx.x covers output columns, blockIdx.y covers
/// output rows.  This matches the spatial layout naturally.
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

/// For max-pooling the gradient rule is simple: the gradient flows *only*
/// to the element that was selected as the maximum in the forward pass.
/// All other input positions within the window receive zero gradient.
///
/// Why atomicAdd?  When pooling windows overlap (stride < pool size) two
/// different output positions could name the same input element as their
/// max.  Even with non-overlapping windows, writing to arbitrary (scattered)
/// locations in grad_in means multiple threads *could* target the same
/// address if the same input value won in two windows.  `atomicAdd`
/// guarantees correct accumulation regardless.
///
/// IMPORTANT: grad_in must be zeroed before calling this kernel.
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

/// Average pooling replaces the max with a simple arithmetic mean over the
/// window.  It treats every input element equally, so the gradient in the
/// backward pass is simply distributed uniformly (1 / pool_area per element).
///
/// Avg-pool is commonly used in the final layers of classification networks
/// ("global average pooling") to collapse spatial dimensions before the
/// fully connected head.
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

/// Because the forward pass computes an average, the gradient is split
/// equally among all inputs inside the window:
///    dL/d(in_{r,c}) += dL/d(out_{or,oc}) / (pool_h * pool_w)
///
/// Like max-pool backward, we use atomicAdd because the scatter targets
/// overlap when stride < pool size (and even without overlap, the write
/// pattern is irregular from the GPU's perspective).
///
/// IMPORTANT: grad_in must be zeroed before calling this kernel.
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
