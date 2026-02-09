/**
 * @file pooling_test.cu
 * @brief Unit tests for Lesson 15 — Max and Average Pooling.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) FAIL() << "CUDA error: " << cudaGetErrorString(err_); \
  } while (0)

// Kernel definitions (self-contained for single-TU compilation) ---------------

__global__ void maxpool_forward(const float* in, float* out, int* indices, int H, int W, int pool_h,
                                int pool_w, int stride) {
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ < OH && oc < OW) {
    float max_val = -1e30F;
    int max_idx = 0;
    for (int ph = 0; ph < pool_h; ++ph)
      for (int pw = 0; pw < pool_w; ++pw) {
        int ir = or_ * stride + ph;
        int ic = oc * stride + pw;
        float val = in[ir * W + ic];
        if (val > max_val) {
          max_val = val;
          max_idx = ir * W + ic;
        }
      }
    int out_idx = or_ * OW + oc;
    out[out_idx] = max_val;
    indices[out_idx] = max_idx;
  }
}

__global__ void maxpool_backward(const float* grad_out, const int* indices, float* grad_in,
                                 int out_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < out_size) atomicAdd(&grad_in[indices[idx]], grad_out[idx]);
}

__global__ void avgpool_forward(const float* in, float* out, int H, int W, int pool_h, int pool_w,
                                int stride) {
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ < OH && oc < OW) {
    float sum = 0.0F;
    for (int ph = 0; ph < pool_h; ++ph)
      for (int pw = 0; pw < pool_w; ++pw) sum += in[(or_ * stride + ph) * W + (oc * stride + pw)];
    out[or_ * OW + oc] = sum / static_cast<float>(pool_h * pool_w);
  }
}

__global__ void avgpool_backward(const float* grad_out, float* grad_in, int H, int W, int pool_h,
                                 int pool_w, int stride) {
  int OH = (H - pool_h) / stride + 1;
  int OW = (W - pool_w) / stride + 1;
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  int or_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (or_ < OH && oc < OW) {
    float grad = grad_out[or_ * OW + oc] / static_cast<float>(pool_h * pool_w);
    for (int ph = 0; ph < pool_h; ++ph)
      for (int pw = 0; pw < pool_w; ++pw)
        atomicAdd(&grad_in[(or_ * stride + ph) * W + (oc * stride + pw)], grad);
  }
}

// Helpers ---------------------------------------------------------------------

struct PoolParams {
  int H, W, pool_h, pool_w, stride;
};

std::ostream& operator<<(std::ostream& os, const PoolParams& p) {
  return os << p.H << "x" << p.W << "_p" << p.pool_h << "x" << p.pool_w << "_s" << p.stride;
}

class MaxPoolTest : public ::testing::TestWithParam<PoolParams> {};
class AvgPoolTest : public ::testing::TestWithParam<PoolParams> {};

static dim3 pool_grid(int OH, int OW) {
  return {static_cast<unsigned>((OW + 15) / 16), static_cast<unsigned>((OH + 15) / 16)};
}
static constexpr dim3 kBlock{16, 16};

// =============================================================================
// Max Pool — forward
// =============================================================================

TEST_P(MaxPoolTest, Forward) {
  auto [H, W, ph, pw, stride] = GetParam();
  int OH = (H - ph) / stride + 1;
  int OW = (W - pw) / stride + 1;

  std::vector<float> h_in(static_cast<size_t>(H) * W);
  for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  int* d_idx;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(OH) * OW * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_idx, static_cast<size_t>(OH) * OW * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

  maxpool_forward<<<pool_grid(OH, OW), kBlock>>>(d_in, d_out, d_idx, H, W, ph, pw, stride);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(static_cast<size_t>(OH) * OW);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference
  for (int r = 0; r < OH; ++r) {
    for (int c = 0; c < OW; ++c) {
      float mx = -1e30F;
      for (int pr = 0; pr < ph; ++pr)
        for (int pc = 0; pc < pw; ++pc)
          mx = std::fmax(mx, h_in[static_cast<size_t>((r * stride + pr) * W + c * stride + pc)]);
      EXPECT_FLOAT_EQ(h_out[static_cast<size_t>(r) * OW + c], mx) << "r=" << r << " c=" << c;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_idx));
}

// =============================================================================
// Max Pool — backward (gradient routing via stored indices)
// =============================================================================

TEST_P(MaxPoolTest, Backward) {
  auto [H, W, ph, pw, stride] = GetParam();
  int OH = (H - ph) / stride + 1;
  int OW = (W - pw) / stride + 1;
  size_t in_size = static_cast<size_t>(H) * W;
  size_t out_size = static_cast<size_t>(OH) * OW;

  std::vector<float> h_in(in_size);
  for (size_t i = 0; i < in_size; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out, *d_grad_out, *d_grad_in;
  int* d_idx;
  CUDA_CHECK(cudaMalloc(&d_in, in_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_idx, out_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_grad_out, out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_in, in_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_grad_in, 0, in_size * sizeof(float)));

  // Forward (to compute indices)
  maxpool_forward<<<pool_grid(OH, OW), kBlock>>>(d_in, d_out, d_idx, H, W, ph, pw, stride);

  // All-ones upstream gradient
  std::vector<float> h_grad_out(out_size, 1.0F);
  CUDA_CHECK(
      cudaMemcpy(d_grad_out, h_grad_out.data(), out_size * sizeof(float), cudaMemcpyHostToDevice));

  int bsz = 256;
  int nblk = (static_cast<int>(out_size) + bsz - 1) / bsz;
  maxpool_backward<<<nblk, bsz>>>(d_grad_out, d_idx, d_grad_in, static_cast<int>(out_size));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad_in(in_size);
  CUDA_CHECK(
      cudaMemcpy(h_grad_in.data(), d_grad_in, in_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Sum of backward grads should equal #(output elements)  (each output routes 1.0)
  float total = std::accumulate(h_grad_in.begin(), h_grad_in.end(), 0.0F);
  EXPECT_NEAR(total, static_cast<float>(out_size), 1e-4F);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_idx));
  CUDA_CHECK(cudaFree(d_grad_out));
  CUDA_CHECK(cudaFree(d_grad_in));
}

// =============================================================================
// Average Pool — forward
// =============================================================================

TEST_P(AvgPoolTest, Forward) {
  auto [H, W, ph, pw, stride] = GetParam();
  int OH = (H - ph) / stride + 1;
  int OW = (W - pw) / stride + 1;
  size_t in_size = static_cast<size_t>(H) * W;
  size_t out_size = static_cast<size_t>(OH) * OW;

  std::vector<float> h_in(in_size);
  for (size_t i = 0; i < in_size; ++i) h_in[i] = static_cast<float>(i) * 0.1F;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, in_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));

  avgpool_forward<<<pool_grid(OH, OW), kBlock>>>(d_in, d_out, H, W, ph, pw, stride);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(out_size);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

  float area = static_cast<float>(ph * pw);
  for (int r = 0; r < OH; ++r) {
    for (int c = 0; c < OW; ++c) {
      float sum = 0.0F;
      for (int pr = 0; pr < ph; ++pr)
        for (int pc = 0; pc < pw; ++pc)
          sum += h_in[static_cast<size_t>((r * stride + pr) * W + c * stride + pc)];
      EXPECT_NEAR(h_out[static_cast<size_t>(r) * OW + c], sum / area, 1e-4F)
          << "r=" << r << " c=" << c;
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

// =============================================================================
// Average Pool — backward (gradient check)
// =============================================================================

TEST_P(AvgPoolTest, BackwardGradientCheck) {
  auto [H, W, ph, pw, stride] = GetParam();
  int OH = (H - ph) / stride + 1;
  int OW = (W - pw) / stride + 1;
  size_t in_size = static_cast<size_t>(H) * W;
  size_t out_size = static_cast<size_t>(OH) * OW;

  std::vector<float> h_in(in_size);
  for (size_t i = 0; i < in_size; ++i) h_in[i] = static_cast<float>(i) * 0.1F;

  float *d_in, *d_out, *d_grad_out, *d_grad_in;
  CUDA_CHECK(cudaMalloc(&d_in, in_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_out, out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_in, in_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_grad_in, 0, in_size * sizeof(float)));

  std::vector<float> h_grad_out(out_size, 1.0F);
  CUDA_CHECK(
      cudaMemcpy(d_grad_out, h_grad_out.data(), out_size * sizeof(float), cudaMemcpyHostToDevice));

  avgpool_backward<<<pool_grid(OH, OW), kBlock>>>(d_grad_out, d_grad_in, H, W, ph, pw, stride);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_grad_in(in_size);
  CUDA_CHECK(
      cudaMemcpy(h_grad_in.data(), d_grad_in, in_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Finite-difference gradient check on a subset
  constexpr float eps = 1e-3F;
  constexpr int check_count = 10;
  int step = std::max(1, static_cast<int>(in_size) / check_count);
  for (int i = 0; i < static_cast<int>(in_size); i += step) {
    auto run_fwd = [&](std::vector<float>& inp) -> float {
      float* d_tmp = nullptr;
      float* d_otmp = nullptr;
      cudaMalloc(&d_tmp, in_size * sizeof(float));
      cudaMalloc(&d_otmp, out_size * sizeof(float));
      cudaMemcpy(d_tmp, inp.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
      avgpool_forward<<<pool_grid(OH, OW), kBlock>>>(d_tmp, d_otmp, H, W, ph, pw, stride);
      cudaDeviceSynchronize();
      std::vector<float> otmp(out_size);
      cudaMemcpy(otmp.data(), d_otmp, out_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(d_tmp);
      cudaFree(d_otmp);
      float s = 0.0F;
      for (auto v : otmp) s += v;
      return s;
    };
    auto inp_p = h_in;
    auto inp_m = h_in;
    inp_p[static_cast<size_t>(i)] += eps;
    inp_m[static_cast<size_t>(i)] -= eps;
    float fd = (run_fwd(inp_p) - run_fwd(inp_m)) / (2.0F * eps);
    EXPECT_NEAR(h_grad_in[static_cast<size_t>(i)], fd, 1e-2F) << "i=" << i;
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_grad_out));
  CUDA_CHECK(cudaFree(d_grad_in));
}

// =============================================================================
// Parameterised Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(Pooling, MaxPoolTest,
                         ::testing::Values(PoolParams{4, 4, 2, 2, 2}, PoolParams{8, 8, 2, 2, 2},
                                           PoolParams{6, 6, 3, 3, 3}, PoolParams{8, 8, 2, 2, 1}));

INSTANTIATE_TEST_SUITE_P(Pooling, AvgPoolTest,
                         ::testing::Values(PoolParams{4, 4, 2, 2, 2}, PoolParams{8, 8, 2, 2, 2},
                                           PoolParams{6, 6, 3, 3, 3}, PoolParams{8, 8, 2, 2, 1}));
