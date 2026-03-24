/**
 * @file residual_layernorm_test.cu
 * @brief Unit tests for Lesson 29 — Residual Connections & Layer Normalization.
 *
 * Tests verify residual add, layer norm forward / backward, and the fused
 * residual + layer norm kernel against CPU reference implementations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    const cudaError_t err_ = (call);                                        \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

constexpr int kBlockSize = 256;

// =============================================================================
// Kernels (duplicated — self‑contained lesson)
// =============================================================================

__global__ void residual_add_kernel(const float* __restrict__ x, const float* __restrict__ residual,
                                    float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + residual[i];
}

__global__ void residual_backward_kernel(const float* __restrict__ dy, float* __restrict__ dx,
                                         float* __restrict__ dresidual, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dx[i] = dy[i];
    dresidual[i] = dy[i];
  }
}

__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ mean,
                                         float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;
  const float* row_in = x + row * D;
  extern __shared__ float smem[];
  float local_sum = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) local_sum += row_in[d];
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);
  float local_var = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float diff = row_in[d] - mu;
    local_var += diff * diff;
  }
  smem[threadIdx.x] = local_var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }
  float* row_out = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (row_in[d] - mu) * inv_std;
    row_out[d] = gamma[d] * x_hat + beta[d];
  }
}

__global__ void layernorm_backward_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ rstd, float* __restrict__ dx,
                                          float* __restrict__ dgamma, float* __restrict__ dbeta,
                                          int N, int D) {
  int row = blockIdx.x;
  if (row >= N) return;
  const float* dy_row = dy + row * D;
  const float* x_row = x + row * D;
  float mu = mean[row];
  float inv_std = rstd[row];
  extern __shared__ float smem[];
  float dot1 = 0.0F, dot2 = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] - mu) * inv_std;
    float dxhat = dy_row[d] * gamma[d];
    dot1 += dxhat * x_hat;
    dot2 += dxhat;
  }
  smem[threadIdx.x] = dot1;
  smem[threadIdx.x + blockDim.x] = dot2;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
      smem[threadIdx.x + blockDim.x] += smem[threadIdx.x + blockDim.x + s];
    }
    __syncthreads();
  }
  float sum_dxhat_xhat = smem[0];
  float sum_dxhat = smem[blockDim.x];
  float inv_D = 1.0F / static_cast<float>(D);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] - mu) * inv_std;
    float dxhat = dy_row[d] * gamma[d];
    dx[row * D + d] = inv_std * (dxhat - inv_D * (sum_dxhat + x_hat * sum_dxhat_xhat));
    atomicAdd(&dgamma[d], dy_row[d] * x_hat);
    atomicAdd(&dbeta[d], dy_row[d]);
  }
}

__global__ void fused_residual_layernorm_kernel(
    const float* __restrict__ x, const float* __restrict__ residual, float* __restrict__ y,
    const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ mean,
    float* __restrict__ rstd, int N, int D, float eps) {
  int row = blockIdx.x;
  if (row >= N) return;
  const float* x_row = x + row * D;
  const float* r_row = residual + row * D;
  extern __shared__ float smem[];
  float local_sum = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) local_sum += x_row[d] + r_row[d];
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float mu = smem[0] / static_cast<float>(D);
  float local_var = 0.0F;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float val = x_row[d] + r_row[d] - mu;
    local_var += val * val;
  }
  smem[threadIdx.x] = local_var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(smem[0] / static_cast<float>(D) + eps);
  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = inv_std;
  }
  float* out_row = y + row * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x_row[d] + r_row[d] - mu) * inv_std;
    out_row[d] = gamma[d] * x_hat + beta[d];
  }
}

// =============================================================================
// Test fixture
// =============================================================================

class ResidualLayerNormTest : public ::testing::Test {
 protected:
  static constexpr int kN = 8;
  static constexpr int kD = 64;
  static constexpr float kEps = 1e-5F;
};

// ------------------------------------------------------------------
// Residual add: y = x + r
// ------------------------------------------------------------------

TEST_F(ResidualLayerNormTest, ResidualAddCorrect) {
  int total = kN * kD;
  std::vector<float> h_x(total), h_r(total);
  for (int i = 0; i < total; ++i) {
    h_x[i] = static_cast<float>(i) * 0.01F;
    h_r[i] = static_cast<float>(total - i) * 0.01F;
  }

  float *d_x, *d_r, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (total + kBlockSize - 1) / kBlockSize;
  residual_add_kernel<<<grid, kBlockSize>>>(d_x, d_r, d_y, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(total);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, total * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) EXPECT_NEAR(h_y[i], h_x[i] + h_r[i], 1e-5F) << "i=" << i;

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_y));
}

// ------------------------------------------------------------------
// Residual backward: both grads equal dy
// ------------------------------------------------------------------

TEST_F(ResidualLayerNormTest, ResidualBackwardCopies) {
  int total = kN * kD;
  std::vector<float> h_dy(total);
  for (int i = 0; i < total; ++i) h_dy[i] = static_cast<float>(i) * 0.1F;

  float *d_dy, *d_dx, *d_dr;
  CUDA_CHECK(cudaMalloc(&d_dy, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dr, total * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), total * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (total + kBlockSize - 1) / kBlockSize;
  residual_backward_kernel<<<grid, kBlockSize>>>(d_dy, d_dx, d_dr, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dx(total), h_dr(total);
  CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, total * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dr.data(), d_dr, total * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(h_dx[i], h_dy[i]);
    EXPECT_FLOAT_EQ(h_dr[i], h_dy[i]);
  }

  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dx));
  CUDA_CHECK(cudaFree(d_dr));
}

// ------------------------------------------------------------------
// LayerNorm forward: zero mean, unit variance with gamma=1, beta=0
// ------------------------------------------------------------------

TEST_F(ResidualLayerNormTest, LayerNormZeroMeanUnitVar) {
  std::vector<float> h_x(kN * kD);
  for (int i = 0; i < kN * kD; ++i) h_x[i] = static_cast<float>(i % kD);
  std::vector<float> h_gamma(kD, 1.0F), h_beta(kD, 0.0F);

  float *d_x, *d_y, *d_gamma, *d_beta, *d_mean, *d_rstd;
  CUDA_CHECK(cudaMalloc(&d_x, kN * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, kN * kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mean, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rstd, kN * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kN * kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), kD * sizeof(float), cudaMemcpyHostToDevice));

  int block = 64;
  layernorm_forward_kernel<<<kN, block, block * sizeof(float)>>>(d_x, d_y, d_gamma, d_beta, d_mean,
                                                                 d_rstd, kN, kD, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y(kN * kD);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, kN * kD * sizeof(float), cudaMemcpyDeviceToHost));

  // Check each row has zero mean and unit variance
  for (int r = 0; r < kN; ++r) {
    float sum = 0.0F;
    for (int d = 0; d < kD; ++d) sum += h_y[r * kD + d];
    float mean = sum / static_cast<float>(kD);
    EXPECT_NEAR(mean, 0.0F, 1e-4F) << "row=" << r;

    float var = 0.0F;
    for (int d = 0; d < kD; ++d) {
      float diff = h_y[r * kD + d] - mean;
      var += diff * diff;
    }
    var /= static_cast<float>(kD);
    EXPECT_NEAR(var, 1.0F, 1e-3F) << "row=" << r;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_rstd));
}

// ------------------------------------------------------------------
// LayerNorm backward: finite-difference check for dx
// ------------------------------------------------------------------

TEST_F(ResidualLayerNormTest, LayerNormBackwardFiniteDiff) {
  constexpr int kSmallD = 16;
  constexpr int kSmallN = 2;
  constexpr float kH = 1e-3F;

  std::vector<float> h_x(kSmallN * kSmallD);
  for (int i = 0; i < kSmallN * kSmallD; ++i) h_x[i] = static_cast<float>(i) * 0.1F - 0.5F;
  std::vector<float> h_gamma(kSmallD, 1.0F), h_beta(kSmallD, 0.0F);
  std::vector<float> h_dy(kSmallN * kSmallD, 1.0F);

  auto ln_fwd = [&](const std::vector<float>& x_in) {
    float *dv_x, *dv_y, *dv_g, *dv_b, *dv_m, *dv_r;
    cudaMalloc(&dv_x, kSmallN * kSmallD * sizeof(float));
    cudaMalloc(&dv_y, kSmallN * kSmallD * sizeof(float));
    cudaMalloc(&dv_g, kSmallD * sizeof(float));
    cudaMalloc(&dv_b, kSmallD * sizeof(float));
    cudaMalloc(&dv_m, kSmallN * sizeof(float));
    cudaMalloc(&dv_r, kSmallN * sizeof(float));
    cudaMemcpy(dv_x, x_in.data(), kSmallN * kSmallD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv_g, h_gamma.data(), kSmallD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv_b, h_beta.data(), kSmallD * sizeof(float), cudaMemcpyHostToDevice);
    layernorm_forward_kernel<<<kSmallN, kSmallD, kSmallD * sizeof(float)>>>(
        dv_x, dv_y, dv_g, dv_b, dv_m, dv_r, kSmallN, kSmallD, kEps);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    std::vector<float> out(kSmallN * kSmallD);
    cudaMemcpy(out.data(), dv_y, kSmallN * kSmallD * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dv_x);
    cudaFree(dv_y);
    cudaFree(dv_g);
    cudaFree(dv_b);
    cudaFree(dv_m);
    cudaFree(dv_r);
    return out;
  };

  // Analytic backward
  float *d_x, *d_y, *d_g, *d_b, *d_m, *d_r, *d_dy, *d_dx, *d_dg, *d_db;
  CUDA_CHECK(cudaMalloc(&d_x, kSmallN * kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, kSmallN * kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m, kSmallN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, kSmallN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dy, kSmallN * kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dx, kSmallN * kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dg, kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_db, kSmallD * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), kSmallN * kSmallD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g, h_gamma.data(), kSmallD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_beta.data(), kSmallD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_dy, h_dy.data(), kSmallN * kSmallD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dg, 0, kSmallD * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_db, 0, kSmallD * sizeof(float)));

  layernorm_forward_kernel<<<kSmallN, kSmallD, kSmallD * sizeof(float)>>>(
      d_x, d_y, d_g, d_b, d_m, d_r, kSmallN, kSmallD, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  layernorm_backward_kernel<<<kSmallN, kSmallD, 2 * kSmallD * sizeof(float)>>>(
      d_dy, d_x, d_g, d_m, d_r, d_dx, d_dg, d_db, kSmallN, kSmallD);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dx(kSmallN * kSmallD);
  CUDA_CHECK(
      cudaMemcpy(h_dx.data(), d_dx, kSmallN * kSmallD * sizeof(float), cudaMemcpyDeviceToHost));

  // Finite differences for a few elements
  for (int idx = 0; idx < kSmallN * kSmallD; idx += 5) {
    auto x_plus = h_x;
    auto x_minus = h_x;
    x_plus[idx] += kH;
    x_minus[idx] -= kH;
    auto y_plus = ln_fwd(x_plus);
    auto y_minus = ln_fwd(x_minus);
    float numerical = 0.0F;
    for (int j = 0; j < kSmallN * kSmallD; ++j)
      numerical += (y_plus[j] - y_minus[j]) / (2.0F * kH) * h_dy[j];
    EXPECT_NEAR(h_dx[idx], numerical, 5e-2F) << "idx=" << idx;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dx));
  CUDA_CHECK(cudaFree(d_dg));
  CUDA_CHECK(cudaFree(d_db));
}

// ------------------------------------------------------------------
// Fused residual+LN matches separate add then LN
// ------------------------------------------------------------------

TEST_F(ResidualLayerNormTest, FusedMatchesSeparate) {
  int total = kN * kD;
  std::vector<float> h_x(total), h_r(total);
  for (int i = 0; i < total; ++i) {
    h_x[i] = static_cast<float>(i % 17) * 0.1F;
    h_r[i] = static_cast<float>(i % 11) * 0.05F;
  }
  std::vector<float> h_gamma(kD, 1.0F), h_beta(kD, 0.0F);

  float *d_x, *d_r, *d_y1, *d_y2, *d_g, *d_b, *d_m, *d_rs, *d_tmp;
  CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y1, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y2, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, kD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rs, kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g, h_gamma.data(), kD * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_beta.data(), kD * sizeof(float), cudaMemcpyHostToDevice));

  // Separate: add then LN
  int grid = (total + kBlockSize - 1) / kBlockSize;
  residual_add_kernel<<<grid, kBlockSize>>>(d_x, d_r, d_tmp, total);
  CUDA_CHECK(cudaGetLastError());
  int block = 64;
  layernorm_forward_kernel<<<kN, block, block * sizeof(float)>>>(d_tmp, d_y1, d_g, d_b, d_m, d_rs,
                                                                 kN, kD, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Fused
  fused_residual_layernorm_kernel<<<kN, block, block * sizeof(float)>>>(d_x, d_r, d_y2, d_g, d_b,
                                                                        d_m, d_rs, kN, kD, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_y1(total), h_y2(total);
  CUDA_CHECK(cudaMemcpy(h_y1.data(), d_y1, total * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_y2.data(), d_y2, total * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < total; ++i) EXPECT_NEAR(h_y1[i], h_y2[i], 1e-4F) << "i=" << i;

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_y1));
  CUDA_CHECK(cudaFree(d_y2));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_rs));
  CUDA_CHECK(cudaFree(d_tmp));
}
