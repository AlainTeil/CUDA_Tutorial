/**
 * @file optimizers_test.cu
 * @brief Unit tests for Lesson 25 — Adam, AdamW & Learning-Rate Schedulers.
 *
 * Tests SGD, Adam, AdamW convergence on the Rosenbrock function, AdamW
 * weight-decay behaviour, and learning-rate scheduler correctness.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                               \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#define CUDA_ASSERT(call)                                                 \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_)); \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

// =============================================================================
// Kernels (duplicated — self-contained lesson)
// =============================================================================

__global__ void sgd_step_kernel(float* __restrict__ param, const float* __restrict__ grad, int n,
                                float lr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) param[i] -= lr * grad[i];
}

__global__ void sgd_momentum_step_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                         float* __restrict__ velocity, int n, float lr, float mu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    velocity[i] = mu * velocity[i] + grad[i];
    param[i] -= lr * velocity[i];
  }
}

__global__ void adam_step_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                 float* __restrict__ m, float* __restrict__ v, int n, float lr,
                                 float beta1, float beta2, float eps, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = grad[i];
  m[i] = beta1 * m[i] + (1.0F - beta1) * g;
  v[i] = beta2 * v[i] + (1.0F - beta2) * g * g;
  float bc1 = 1.0F - powf(beta1, static_cast<float>(t));
  float bc2 = 1.0F - powf(beta2, static_cast<float>(t));
  float m_hat = m[i] / bc1;
  float v_hat = v[i] / bc2;
  param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

__global__ void adamw_step_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                  float* __restrict__ m, float* __restrict__ v, int n, float lr,
                                  float beta1, float beta2, float eps, float wd, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  param[i] -= lr * wd * param[i];
  float g = grad[i];
  m[i] = beta1 * m[i] + (1.0F - beta1) * g;
  v[i] = beta2 * v[i] + (1.0F - beta2) * g * g;
  float bc1 = 1.0F - powf(beta1, static_cast<float>(t));
  float bc2 = 1.0F - powf(beta2, static_cast<float>(t));
  float m_hat = m[i] / bc1;
  float v_hat = v[i] / bc2;
  param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// =============================================================================
// LR schedulers (host-side, duplicated)
// =============================================================================

static float cosine_lr(float eta_max, float eta_min, int t, int T) {
  double ratio = static_cast<double>(t) / static_cast<double>(T);
  return eta_min + 0.5F * (eta_max - eta_min) * (1.0F + static_cast<float>(std::cos(M_PI * ratio)));
}

static float warmup_cosine_lr(float eta_max, float eta_min, int t, int warmup_steps,
                              int total_steps) {
  if (t < warmup_steps) return eta_max * static_cast<float>(t) / static_cast<float>(warmup_steps);
  int ds = t - warmup_steps;
  int dt = total_steps - warmup_steps;
  return cosine_lr(eta_max, eta_min, ds, dt);
}

// ---- Rosenbrock helper ------------------------------------------------------

static void rosenbrock_grad(float x, float y, float& gx, float& gy) {
  gx = -2.0F * (1.0F - x) + 200.0F * (y - x * x) * (-2.0F * x);
  gy = 200.0F * (y - x * x);
}

static float rosenbrock_loss(float x, float y) {
  return (1.0F - x) * (1.0F - x) + 100.0F * (y - x * x) * (y - x * x);
}

// ---- Tests -----------------------------------------------------------------

TEST(OptimizerTest, SGDConverges) {
  float *d_p, *d_g;
  CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
  float p[2] = {-1.0F, -1.0F};
  CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));

  float initial = rosenbrock_loss(p[0], p[1]);
  for (int t = 0; t < 5000; ++t) {
    float g[2];
    rosenbrock_grad(p[0], p[1], g[0], g[1]);
    CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
    sgd_step_kernel<<<1, 2>>>(d_p, d_g, 2, 1e-3F);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  }
  float final_loss = rosenbrock_loss(p[0], p[1]);
  EXPECT_LT(final_loss, initial * 0.1F) << "SGD should reduce Rosenbrock loss significantly";

  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_g));
}

TEST(OptimizerTest, AdamConvergesOnRosenbrock) {
  // Adam should converge to near the Rosenbrock minimum (1,1).
  float *d_p, *d_g, *d_m, *d_v;
  CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));
  float p[2] = {-1.0F, -1.0F};
  CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_m, 0, 2 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

  float initial = rosenbrock_loss(p[0], p[1]);
  for (int t = 1; t <= 20000; ++t) {
    float g[2];
    rosenbrock_grad(p[0], p[1], g[0], g[1]);
    CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
    adam_step_kernel<<<1, 2>>>(d_p, d_g, d_m, d_v, 2, 1e-3F, 0.9F, 0.999F, 1e-8F, t);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  }
  float final_loss = rosenbrock_loss(p[0], p[1]);
  EXPECT_LT(final_loss, initial * 0.01F) << "Adam should significantly reduce Rosenbrock loss";
  EXPECT_LT(final_loss, 1.0F) << "Adam should converge near the minimum";

  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_v));
}

TEST(OptimizerTest, AdamWWeightDecay) {
  // With zero gradient, AdamW should still shrink weights due to weight decay
  float *d_p, *d_g, *d_m, *d_v;
  CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));

  float p[2] = {5.0F, 5.0F};
  CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_g, 0, 2 * sizeof(float)));  // zero gradient
  CUDA_CHECK(cudaMemset(d_m, 0, 2 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

  float lr = 0.01F;
  float wd = 0.1F;
  adamw_step_kernel<<<1, 2>>>(d_p, d_g, d_m, d_v, 2, lr, 0.9F, 0.999F, 1e-8F, wd, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));

  // Weight should decrease: 5.0 - 0.01 * 0.1 * 5.0 = 4.995
  // (the Adam step with zero gradient is essentially zero due to m=v=0+eps)
  EXPECT_LT(p[0], 5.0F) << "Weight decay should shrink parameters";
  EXPECT_LT(p[1], 5.0F);

  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_v));
}

TEST(OptimizerTest, CosineLRSchedule) {
  float lr_start = cosine_lr(1e-3F, 1e-5F, 0, 100);
  float lr_mid = cosine_lr(1e-3F, 1e-5F, 50, 100);
  float lr_end = cosine_lr(1e-3F, 1e-5F, 100, 100);

  EXPECT_NEAR(lr_start, 1e-3F, 1e-7F) << "LR at t=0 should be η_max";
  EXPECT_NEAR(lr_end, 1e-5F, 1e-7F) << "LR at t=T should be η_min";
  // Midpoint: η_min + 0.5*(η_max-η_min)*(1+cos(π/2)) = η_min + 0.5*(η_max-η_min)
  float expected_mid = 1e-5F + 0.5F * (1e-3F - 1e-5F);
  EXPECT_NEAR(lr_mid, expected_mid, 1e-6F) << "LR at t=T/2 should be midpoint";
}

TEST(OptimizerTest, WarmupCosineLR) {
  // During warmup: linear increase
  float lr0 = warmup_cosine_lr(1e-3F, 1e-5F, 0, 10, 100);
  float lr5 = warmup_cosine_lr(1e-3F, 1e-5F, 5, 10, 100);
  float lr10 = warmup_cosine_lr(1e-3F, 1e-5F, 10, 10, 100);

  EXPECT_NEAR(lr0, 0.0F, 1e-7F) << "LR at t=0 with warmup should be 0";
  EXPECT_NEAR(lr5, 0.5e-3F, 1e-7F) << "LR at t=warmup/2 should be η_max/2";
  EXPECT_NEAR(lr10, 1e-3F, 1e-6F) << "LR at warmup end should be η_max";

  // After warmup: cosine decay
  float lr_last = warmup_cosine_lr(1e-3F, 1e-5F, 100, 10, 100);
  EXPECT_NEAR(lr_last, 1e-5F, 1e-6F) << "LR at final step should be η_min";
}

TEST(OptimizerTest, MomentumAccelerates) {
  float *d_p1, *d_g1, *d_p2, *d_g2, *d_v;
  CUDA_CHECK(cudaMalloc(&d_p1, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g1, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_p2, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g2, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));

  float p1[2] = {-1.0F, -1.0F};
  float p2[2] = {-1.0F, -1.0F};
  CUDA_CHECK(cudaMemcpy(d_p1, p1, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_p2, p2, 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

  for (int t = 0; t < 5000; ++t) {
    float g1[2], g2[2];
    rosenbrock_grad(p1[0], p1[1], g1[0], g1[1]);
    rosenbrock_grad(p2[0], p2[1], g2[0], g2[1]);
    CUDA_CHECK(cudaMemcpy(d_g1, g1, 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2, g2, 2 * sizeof(float), cudaMemcpyHostToDevice));
    sgd_step_kernel<<<1, 2>>>(d_p1, d_g1, 2, 5e-4F);
    CUDA_CHECK(cudaGetLastError());
    sgd_momentum_step_kernel<<<1, 2>>>(d_p2, d_g2, d_v, 2, 5e-4F, 0.9F);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(p1, d_p1, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p2, d_p2, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  }

  float loss_sgd = rosenbrock_loss(p1[0], p1[1]);
  float loss_mom = rosenbrock_loss(p2[0], p2[1]);
  EXPECT_LT(loss_mom, loss_sgd) << "Momentum should converge to lower loss than vanilla SGD";

  CUDA_CHECK(cudaFree(d_p1));
  CUDA_CHECK(cudaFree(d_g1));
  CUDA_CHECK(cudaFree(d_p2));
  CUDA_CHECK(cudaFree(d_g2));
  CUDA_CHECK(cudaFree(d_v));
}
