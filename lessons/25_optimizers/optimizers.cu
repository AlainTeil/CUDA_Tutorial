/**
 * @file optimizers.cu
 * @brief Lesson 25 — Adam, AdamW & Learning-Rate Schedulers.
 *
 * Lesson 17 used **vanilla SGD** (`θ ← θ − lr·g`) which has two well-known
 * problems:
 *
 *   1. It uses a single global learning rate for every parameter, even
 *      though different parameters may have vastly different gradient
 *      magnitudes.
 *   2. It has no "memory" — the update at step t depends only on the
 *      gradient at step t, ignoring past history.
 *
 * ## SGD with Momentum
 *
 * Momentum maintains an exponential moving average of past gradients:
 *
 *     v_t = μ · v_{t-1} + g_t            (velocity)
 *     θ_t = θ_{t-1} − lr · v_t
 *
 * This accelerates convergence in narrow valleys by accumulating velocity
 * in consistent gradient directions while damping oscillations.
 *
 * ## Adam (Adaptive Moment Estimation)
 *
 * Adam (Kingma & Ba, 2015) combines momentum with per-parameter adaptive
 * learning rates by tracking both the first moment (mean) **m** and second
 * moment (uncentred variance) **v** of the gradients:
 *
 *     m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
 *     v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
 *     m̂_t = m_t / (1 − β₁ᵗ)               (bias correction)
 *     v̂_t = v_t / (1 − β₂ᵗ)
 *     θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
 *
 * ## AdamW (Decoupled Weight Decay)
 *
 * Loshchilov & Hutter (2019) showed that L2 regularisation with Adam is
 * **not** equivalent to weight decay.  AdamW applies weight decay directly
 * to the parameters, *outside* the adaptive gradient step:
 *
 *     θ_t = θ_{t-1} − lr · (m̂_t / (√v̂_t + ε) + wd · θ_{t-1})
 *
 * ## Learning-Rate Schedulers
 *
 * A common practice is to **warm up** the learning rate linearly for the
 * first few steps (to avoid large early updates), then **decay** it
 * following a cosine schedule:
 *
 *     Cosine:         η_t  = η_min + 0.5·(η_max − η_min)·(1 + cos(πt/T))
 *     Warmup+Cosine:  η_t  = η_max · t / T_warmup         (t < T_warmup)
 *                     η_t  = cosine schedule thereafter
 *
 * See Lesson 17 for the SGD baseline and Lesson 32 (Transformer capstone)
 * for an Adam + warmup-cosine training pipeline.
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
// Optimiser kernels
// =============================================================================

/**
 * @brief Vanilla SGD: θ ← θ − lr·g.
 *
 * @param param  Model parameters (updated in place).
 * @param grad   Parameter gradients.
 * @param n      Number of elements.
 * @param lr     Learning rate.
 */
__global__ void sgd_step_kernel(float* __restrict__ param, const float* __restrict__ grad, int n,
                                float lr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) param[i] -= lr * grad[i];
}

/**
 * @brief SGD with Nesterov-style momentum.
 *
 * v_t = μ·v_{t-1} + g_t
 * θ_t = θ_{t-1} − lr·v_t
 *
 * @param param     Model parameters (updated in place).
 * @param grad      Parameter gradients.
 * @param velocity  Momentum velocity buffer (updated in place).
 * @param n         Number of elements.
 * @param lr        Learning rate.
 * @param mu        Momentum coefficient.
 */
__global__ void sgd_momentum_step_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                         float* __restrict__ velocity, int n, float lr, float mu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    velocity[i] = mu * velocity[i] + grad[i];
    param[i] -= lr * velocity[i];
  }
}

/**
 * @brief Adam optimiser step (one element per thread).
 *
 * m, v are per-parameter moment buffers; t is the 1-based timestep.
 * All are updated in place.
 *
 * @param param  Model parameters (updated in place).
 * @param grad   Parameter gradients.
 * @param m      First moment estimate buffer (updated in place).
 * @param v      Second moment estimate buffer (updated in place).
 * @param n      Number of elements.
 * @param lr     Learning rate.
 * @param beta1  Exponential decay rate for the first moment.
 * @param beta2  Exponential decay rate for the second moment.
 * @param eps    Small constant for numerical stability.
 * @param t      Current 1-based timestep for bias correction.
 */
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

/**
 * @brief AdamW: Adam with decoupled weight decay.
 *
 * The weight decay term `lr * wd * θ` is applied to the parameter
 * directly, *separate* from the adaptive gradient step.
 *
 * @param param  Model parameters (updated in place).
 * @param grad   Parameter gradients.
 * @param m      First moment estimate buffer (updated in place).
 * @param v      Second moment estimate buffer (updated in place).
 * @param n      Number of elements.
 * @param lr     Learning rate.
 * @param beta1  Exponential decay rate for the first moment.
 * @param beta2  Exponential decay rate for the second moment.
 * @param eps    Small constant for numerical stability.
 * @param wd     Weight decay coefficient.
 * @param t      Current 1-based timestep for bias correction.
 */
__global__ void adamw_step_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                  float* __restrict__ m, float* __restrict__ v, int n, float lr,
                                  float beta1, float beta2, float eps, float wd, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Weight decay (decoupled)
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
// Learning-rate schedulers (host-side)
// =============================================================================

/// @brief Cosine-annealing learning rate.
///
/// Returns η_min + 0.5 · (η_max − η_min) · (1 + cos(π · t / T)).
float cosine_lr(float eta_max, float eta_min, int t, int T) {
  double ratio = static_cast<double>(t) / static_cast<double>(T);
  return eta_min + 0.5F * (eta_max - eta_min) * (1.0F + static_cast<float>(std::cos(M_PI * ratio)));
}

/// @brief Linear warmup followed by cosine decay.
float warmup_cosine_lr(float eta_max, float eta_min, int t, int warmup_steps, int total_steps) {
  if (t < warmup_steps) {
    return eta_max * static_cast<float>(t) / static_cast<float>(warmup_steps);
  }
  int decay_step = t - warmup_steps;
  int decay_total = total_steps - warmup_steps;
  return cosine_lr(eta_max, eta_min, decay_step, decay_total);
}

// =============================================================================
// Rosenbrock gradient (CPU helper for minimisation demo)
// =============================================================================

/// @brief Rosenbrock function: f(x,y) = (1−x)² + 100·(y−x²)².
/// Minimum at (1, 1) with f = 0.
static void rosenbrock_grad(float x, float y, float& gx, float& gy) {
  gx = -2.0F * (1.0F - x) + 200.0F * (y - x * x) * (-2.0F * x);
  gy = 200.0F * (y - x * x);
}

// =============================================================================
// main — compare optimisers on the Rosenbrock function
// =============================================================================

int main() {
  constexpr int kSteps = 5000;
  constexpr float kLR = 1e-3F;

  // ---- SGD ------------------------------------------------------------------
  {
    float *d_p, *d_g;
    CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
    float p[2] = {-1.0F, -1.0F};
    CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));

    for (int t = 0; t < kSteps; ++t) {
      float g[2];
      rosenbrock_grad(p[0], p[1], g[0], g[1]);
      CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
      sgd_step_kernel<<<1, 2>>>(d_p, d_g, 2, kLR);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    float loss =
        (1.0F - p[0]) * (1.0F - p[0]) + 100.0F * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
    std::printf("SGD          : (%.6f, %.6f)  loss = %.8f\n", static_cast<double>(p[0]),
                static_cast<double>(p[1]), static_cast<double>(loss));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_g));
  }

  // ---- SGD + Momentum -------------------------------------------------------
  {
    float *d_p, *d_g, *d_v;
    CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));
    float p[2] = {-1.0F, -1.0F};
    CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

    for (int t = 0; t < kSteps; ++t) {
      float g[2];
      rosenbrock_grad(p[0], p[1], g[0], g[1]);
      CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
      sgd_momentum_step_kernel<<<1, 2>>>(d_p, d_g, d_v, 2, kLR, 0.9F);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    float loss =
        (1.0F - p[0]) * (1.0F - p[0]) + 100.0F * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
    std::printf("SGD+Momentum : (%.6f, %.6f)  loss = %.8f\n", static_cast<double>(p[0]),
                static_cast<double>(p[1]), static_cast<double>(loss));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_v));
  }

  // ---- Adam -----------------------------------------------------------------
  {
    float *d_p, *d_g, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));
    float p[2] = {-1.0F, -1.0F};
    CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 0, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

    for (int t = 1; t <= kSteps; ++t) {
      float g[2];
      rosenbrock_grad(p[0], p[1], g[0], g[1]);
      CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
      adam_step_kernel<<<1, 2>>>(d_p, d_g, d_m, d_v, 2, kLR, 0.9F, 0.999F, 1e-8F, t);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    float loss =
        (1.0F - p[0]) * (1.0F - p[0]) + 100.0F * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
    std::printf("Adam         : (%.6f, %.6f)  loss = %.8f\n", static_cast<double>(p[0]),
                static_cast<double>(p[1]), static_cast<double>(loss));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_v));
  }

  // ---- AdamW ----------------------------------------------------------------
  {
    float *d_p, *d_g, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_p, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, 2 * sizeof(float)));
    float p[2] = {-1.0F, -1.0F};
    CUDA_CHECK(cudaMemcpy(d_p, p, 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 0, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, 2 * sizeof(float)));

    for (int t = 1; t <= kSteps; ++t) {
      float g[2];
      rosenbrock_grad(p[0], p[1], g[0], g[1]);
      CUDA_CHECK(cudaMemcpy(d_g, g, 2 * sizeof(float), cudaMemcpyHostToDevice));
      adamw_step_kernel<<<1, 2>>>(d_p, d_g, d_m, d_v, 2, kLR, 0.9F, 0.999F, 1e-8F, 0.01F, t);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    float loss =
        (1.0F - p[0]) * (1.0F - p[0]) + 100.0F * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]);
    std::printf("AdamW(wd=.01): (%.6f, %.6f)  loss = %.8f\n", static_cast<double>(p[0]),
                static_cast<double>(p[1]), static_cast<double>(loss));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_v));
  }

  // ---- Learning-rate schedule ------------------------------------------------
  std::printf("\nWarmup-Cosine LR schedule (100 steps, 10 warmup):\n");
  for (int t = 0; t <= 100; t += 10) {
    float lr = warmup_cosine_lr(1e-3F, 1e-5F, t, 10, 100);
    std::printf("  step %3d → lr = %.6f\n", t, static_cast<double>(lr));
  }

  return EXIT_SUCCESS;
}
