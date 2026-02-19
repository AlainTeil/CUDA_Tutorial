/**
 * @file mixed_precision.cu
 * @brief Lesson 21 — Mixed-Precision Training (FP16 / TF32).
 *
 * Modern GPUs (Volta and newer) have **Tensor Cores** that can perform
 * matrix-multiply-accumulate operations much faster than their FP32 CUDA
 * cores — but they operate on reduced-precision inputs (FP16, BF16, TF32).
 *
 * **Mixed-precision training** exploits this by:
 *
 *   1. Storing weights and activations in **FP16** (`__half`) to halve
 *      memory traffic.
 *   2. Performing matrix multiplies via cuBLAS `cublasGemmEx` which can
 *      use **Tensor Cores** for the heavy arithmetic.
 *   3. Keeping a **master copy of weights in FP32** and accumulating
 *      gradients in FP32 to preserve numerical stability.
 *   4. Using **loss scaling** to prevent FP16 gradients from underflowing
 *      to zero.
 *
 * This lesson also demonstrates **TF32** mode, where FP32 data is
 * transparently rounded to 19-bit mantissa before entering the Tensor
 * Cores — giving a ~4× speed-up over FP32 CUDA cores with almost no
 * accuracy loss and zero code changes.
 *
 * ## Precision formats compared
 *
 * | Format | Bits | Exponent | Mantissa | Tensor Core | Notes               |
 * |--------|------|----------|----------|-------------|---------------------|
 * | FP32   | 32   | 8        | 23       | TF32 path   | Full precision      |
 * | TF32   | 19   | 8        | 10       | Yes         | Internal to TC      |
 * | FP16   | 16   | 5        | 10       | Yes         | Needs loss scaling  |
 * | BF16   | 16   | 8        | 7        | Ampere+     | Wider range than FP16|
 *
 * ## Architecture
 *
 * Same 2-layer MLP as Lesson 20:
 *   X[B×4] → Dense(16) → ReLU → Dense(3) → Softmax+CE → loss
 *
 * But now with three precision modes:
 *   - **FP32 baseline** — cuBLAS SGEMM (no Tensor Cores)
 *   - **TF32 mode**     — cuBLAS SGEMM with TF32 Tensor Cores enabled
 *   - **FP16 mode**     — cuBLAS GemmEx with FP16 I/O, FP32 accumulation
 *
 * ## Key APIs introduced
 *
 *   - `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` — enable TF32
 *   - `cublasGemmEx(...)` — type-flexible GEMM supporting mixed types
 *   - `__half`, `__float2half`, `__half2float` — FP16 type and conversions
 *   - `cuda_fp16.h` — FP16 arithmetic header
 *
 * Build: requires cuBLAS (always available with the CUDA Toolkit).
 *
 * @note Tensor Cores require sm_70+ (Volta).  TF32 requires sm_80+ (Ampere).
 *       FP16 Tensor Cores are available on sm_70+.
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// =============================================================================
// Error-checking macros
// =============================================================================

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

// =============================================================================
// FP16 ↔ FP32 conversion kernels
// =============================================================================

/// @defgroup fp16_convert FP16 ↔ FP32 conversion
/// @{

/// Convert an array of FP32 values to FP16 (`__half`).
///
/// Each thread converts one element.  The `__float2half()` intrinsic rounds
/// the 32-bit float to the nearest representable 16-bit value.  Values
/// outside FP16 range (±65504) saturate to ±inf.
__global__ void fp32_to_fp16_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = __float2half(src[i]);
  }
}

/// Convert an array of FP16 values back to FP32.
///
/// This is lossless — every FP16 value is exactly representable in FP32.
__global__ void fp16_to_fp32_kernel(const __half* __restrict__ src, float* __restrict__ dst,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = __half2float(src[i]);
  }
}

/// @}

/// Helper: launch FP32 → FP16 conversion.
static void fp32_to_fp16(const float* d_src, __half* d_dst, int n) {
  int blk = 256;
  fp32_to_fp16_kernel<<<(n + blk - 1) / blk, blk>>>(d_src, d_dst, n);
}

/// Helper: launch FP16 → FP32 conversion.
static void fp16_to_fp32(const __half* d_src, float* d_dst, int n) {
  int blk = 256;
  fp16_to_fp32_kernel<<<(n + blk - 1) / blk, blk>>>(d_src, d_dst, n);
}

// =============================================================================
// cuBLAS GEMM wrappers — three precision modes
// =============================================================================

/// @defgroup gemm_wrappers GEMM wrappers for different precisions
/// @{

/// **FP32 GEMM** (row-major): C[M×N] = α·A[M×K]·B[K×N] + β·C[M×N].
///
/// Uses the standard cuBLAS transpose trick for row-major layout:
///   C_rm = A_rm · B_rm  ←→  C^T_cm = B^T_cm · A^T_cm
/// We swap A↔B and M↔N in the cuBLAS call.
///
/// With `CUBLAS_DEFAULT_MATH`, this runs on CUDA cores (pure FP32).
/// With `CUBLAS_TF32_TENSOR_OP_MATH`, the same call transparently uses
/// Tensor Cores in TF32 mode on Ampere+ GPUs — no API change needed.
static void gemm_fp32(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                      const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/// **FP16 GEMM** (row-major): C_fp16[M×N] = α·A_fp16[M×K]·B_fp16[K×N] + β·C_fp16.
///
/// Uses `cublasGemmEx` with:
///   - Input type: `CUDA_R_16F` (FP16)
///   - Output type: `CUDA_R_16F` (FP16)
///   - Compute type: `CUBLAS_COMPUTE_32F` (FP32 accumulation)
///
/// FP32 accumulation is critical: FP16 has only ~3 decimal digits of
/// precision, so accumulating a sum of K products in FP16 would lose
/// accuracy rapidly.  Tensor Cores natively accumulate in FP32.
///
/// The row-major trick is the same — swap A↔B and M↔N.
static void gemm_fp16(cublasHandle_t h, int M, int K, int N, float alpha, const __half* A,
                      const __half* B, float beta, __half* C) {
  // Row-major trick: swap A↔B, M↔N.
  // cuBLAS sees: C^T[N×M] = B^T[N×K] · A^T[K×M],  OP_N on both since
  // row-major storage already looks like the transpose in column-major.
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,  // m_cublas, n_cublas, k
                            &alpha,                                // host pointer, FP32
                            B, CUDA_R_16F, N,                      // A_cublas = our B, lda = N
                            A, CUDA_R_16F, K,                      // B_cublas = our A, ldb = K
                            &beta,                                 // host pointer, FP32
                            C, CUDA_R_16F, N,                      // C, ldc = N
                            CUBLAS_COMPUTE_32F,                    // accumulate in FP32
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

/// **FP16 GEMM with transposed A** (row-major):
///   C_fp16[M×N] = α · A^T_fp16[M×K] · B_fp16[K×N] + β · C_fp16
///
/// A is stored as [K×M] row-major.  Used for dW = X^T · dZ.
static void gemm_fp16_AT(cublasHandle_t h, int M, int K, int N, float alpha, const __half* A,
                         const __half* B, float beta, __half* C) {
  // Swap world: C^T[N×M] = B^T[N×K] · (A^T)^T[K×M]
  // A stored [K×M] row-major = [M×K] col-major.  We need [K×M] col → OP_T.
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, CUDA_R_16F, N, A,
                            CUDA_R_16F, M, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

/// **FP32 GEMM with transposed A** (row-major):
///   C[M×N] = α · A^T[M×K] · B[K×N] + β · C
///
/// Used in TF32/FP32 mode for dW = X^T · dY.
static void gemm_fp32_AT(cublasHandle_t h, int M, int K, int N, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

/// @}

// =============================================================================
// Element-wise CUDA kernels (FP32 — used for gradient computation)
// =============================================================================

/// @defgroup kernels Element-wise kernels
/// @{

/// ReLU forward: out = max(0, x).
__global__ void relu_fwd(const float* __restrict__ x, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fmaxf(x[i], 0.0F);
  }
}

/// ReLU backward: dX = dY * (Z > 0).
__global__ void relu_bwd(const float* __restrict__ z, const float* __restrict__ dy,
                         float* __restrict__ dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dx[i] = (z[i] > 0.0F) ? dy[i] : 0.0F;
  }
}

/// Add bias: out[b*D + j] += bias[j]  for all rows b in [0, B).
__global__ void add_bias(float* __restrict__ out, const float* __restrict__ bias, int B, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * D) {
    int j = idx % D;
    out[idx] += bias[j];
  }
}

/// Cross-entropy backward: dZ = softmax(logits) − target.
/// Expects `log_sm` = log-softmax output, `target` = one-hot.
__global__ void ce_bwd(const float* __restrict__ log_sm, const float* __restrict__ target,
                       float* __restrict__ dz, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dz[i] = expf(log_sm[i]) - target[i];
  }
}

/// Column sum: out[j] = Σ_b  in[b*D + j],  j ∈ [0, D).
__global__ void col_sum(const float* __restrict__ in, float* __restrict__ out, int B, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) {
    float s = 0.0F;
    for (int b = 0; b < B; ++b) {
      s += in[b * D + j];
    }
    out[j] = s;
  }
}

/// Scale every element by a constant.
__global__ void scale_kernel(float* __restrict__ data, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] *= s;
  }
}

/// Batch log-softmax: one block per row (sample).
/// Computes log_sm[b][c] = logit[b][c] − log(Σ_c exp(logit[b][c])).
__global__ void batch_log_softmax(const float* __restrict__ logits, float* __restrict__ log_sm,
                                  int C) {
  int b = blockIdx.x;
  const float* row = logits + b * C;
  float* out = log_sm + b * C;

  // Numerically stable: subtract max before exp.
  float mx = row[0];
  for (int c = 1; c < C; ++c) mx = fmaxf(mx, row[c]);

  float sum = 0.0F;
  for (int c = 0; c < C; ++c) sum += expf(row[c] - mx);

  float log_sum = logf(sum) + mx;
  for (int c = 0; c < C; ++c) out[c] = row[c] - log_sum;
}

/// Cross-entropy loss from log-softmax: L = −(1/B) Σ_b Σ_c  t[b][c] · ls[b][c].
__global__ void ce_loss_kernel(const float* __restrict__ log_sm, const float* __restrict__ target,
                               float* __restrict__ loss, int B, int C) {
  // Single-thread kernel for simplicity; called with <<<1,1>>>.
  float s = 0.0F;
  for (int i = 0; i < B * C; ++i) {
    s -= target[i] * log_sm[i];
  }
  *loss = s / static_cast<float>(B);
}

/// SGD update: W -= lr * dW.
__global__ void sgd_update(float* __restrict__ W, const float* __restrict__ dW, float lr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    W[i] -= lr * dW[i];
  }
}

/// Grid-stride reduction: sum all elements.
__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  float s = 0.0F;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    s += in[i];
  }

  // Warp-level reduction.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    s += __shfl_down_sync(0xFFFFFFFF, s, offset);
  }

  // First lane of each warp atomically adds to output.
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(out, s);
  }
}

/// @}

// =============================================================================
// Precision mode enum
// =============================================================================

/// Selects which precision path to use for GEMM operations.
///
/// - **FP32**: pure FP32 on CUDA cores (baseline).
/// - **TF32**: FP32 data, but cuBLAS transparently uses Tensor Cores with
///   TF32 rounding (10-bit mantissa instead of 23-bit).  No code changes
///   needed — just flip a math-mode flag.
/// - **FP16**: weights and activations stored in `__half`, GEMM computed
///   via `cublasGemmEx` with FP32 accumulation.  Gradients computed in
///   FP32 and converted back to FP16 for the next iteration.
enum class PrecisionMode { kFP32, kTF32, kFP16 };

// =============================================================================
// MixedPrecisionMLP — the training struct
// =============================================================================

/// @brief 2-layer MLP supporting FP32, TF32, and FP16 training.
///
/// The struct maintains:
///   - **Master weights in FP32** — always the authoritative copy.
///   - **FP16 shadow weights** (if mode == kFP16) — converted from FP32
///     before each forward pass.
///   - **Activations and gradients in FP32** — gradient computation stays
///     in FP32 for numerical stability.
///   - **FP16 activation mirrors** (if mode == kFP16) — converted for
///     the GEMM calls.
///
/// ### Loss scaling (FP16 mode)
///
/// FP16 has limited range (min positive normal ≈ 6.1e-5).  Small gradient
/// values that are perfectly fine in FP32 can underflow to zero in FP16.
/// The fix is **loss scaling**:
///
///   1. Multiply the loss by a large constant S (e.g. 1024) before
///      backward.  This scales all gradients by S.
///   2. After computing dW in FP32, divide by S before the SGD update.
///
/// This keeps small gradients in the representable FP16 range during
/// backpropagation without affecting the final weight update.
struct MixedPrecisionMLP {
  // Network dimensions.
  int in_dim{};
  int hid_dim{};
  int out_dim{};
  int max_batch{};

  PrecisionMode mode{PrecisionMode::kFP32};
  float loss_scale{1.0F};  ///< Loss scaling factor for FP16 mode.

  cublasHandle_t cublas{};

  // FP32 master weights & biases  (always present).
  float *W1{}, *b1{};  ///< Layer 1: W1[in×hid], b1[hid].
  float *W2{}, *b2{};  ///< Layer 2: W2[hid×out], b2[out].

  // FP32 activations & gradients.
  float *Z1{}, *A1{}, *Z2{}, *LS{};      ///< Forward: pre-act, post-act, logits, log-softmax.
  float *dZ2{}, *dA1{}, *dZ1{};          ///< Backward: gradients.
  float *dW1{}, *db1{}, *dW2{}, *db2{};  ///< Weight gradients.

  // FP16 shadows (allocated only in kFP16 mode).
  __half *W1_h{}, *W2_h{};              ///< FP16 copies of weights.
  __half* X_h{};                        ///< FP16 copy of input batch.
  __half *Z1_h{}, *A1_h{};              ///< FP16 layer-1 activation.
  __half* Z2_h{};                       ///< FP16 layer-2 pre-activation.
  __half *dZ2_h{}, *dA1_h{}, *dZ1_h{};  ///< FP16 gradient mirrors.
  __half *dW2_h{}, *dW1_h{};            ///< FP16 weight gradient scratch.

  /// Allocate all device buffers.
  void alloc(int in, int hid, int out, int max_B, PrecisionMode p, float ls = 1024.0F) {
    in_dim = in;
    hid_dim = hid;
    out_dim = out;
    max_batch = max_B;
    mode = p;
    loss_scale = (p == PrecisionMode::kFP16) ? ls : 1.0F;

    CUBLAS_CHECK(cublasCreate(&cublas));

    // --- Set math mode ---
    //
    // CUBLAS_DEFAULT_MATH:          pure FP32 on CUDA cores.
    // CUBLAS_TF32_TENSOR_OP_MATH:   FP32 data, but cuBLAS rounds mantissa
    //   to 10 bits and uses Tensor Cores.  ~4× faster on Ampere+.
    //
    // For kFP16 mode we use cublasGemmEx which selects Tensor Cores via
    // the CUBLAS_GEMM_DEFAULT_TENSOR_OP algo flag, so the handle math
    // mode doesn't matter — but we set it to default for clarity.
    if (mode == PrecisionMode::kTF32) {
      CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    } else {
      CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
    }

    // FP32 buffers — always needed.
    auto alloc_f = [](float** ptr, size_t count) {
      CUDA_CHECK(cudaMalloc(ptr, count * sizeof(float)));
    };
    alloc_f(&W1, static_cast<size_t>(in) * hid);
    alloc_f(&b1, hid);
    alloc_f(&W2, static_cast<size_t>(hid) * out);
    alloc_f(&b2, out);
    alloc_f(&Z1, static_cast<size_t>(max_B) * hid);
    alloc_f(&A1, static_cast<size_t>(max_B) * hid);
    alloc_f(&Z2, static_cast<size_t>(max_B) * out);
    alloc_f(&LS, static_cast<size_t>(max_B) * out);
    alloc_f(&dZ2, static_cast<size_t>(max_B) * out);
    alloc_f(&dA1, static_cast<size_t>(max_B) * hid);
    alloc_f(&dZ1, static_cast<size_t>(max_B) * hid);
    alloc_f(&dW1, static_cast<size_t>(in) * hid);
    alloc_f(&db1, hid);
    alloc_f(&dW2, static_cast<size_t>(hid) * out);
    alloc_f(&db2, out);

    // FP16 buffers — only for kFP16 mode.
    if (mode == PrecisionMode::kFP16) {
      auto alloc_h = [](__half** ptr, size_t count) {
        CUDA_CHECK(cudaMalloc(ptr, count * sizeof(__half)));
      };
      alloc_h(&W1_h, static_cast<size_t>(in) * hid);
      alloc_h(&W2_h, static_cast<size_t>(hid) * out);
      alloc_h(&X_h, static_cast<size_t>(max_B) * in);
      alloc_h(&Z1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&A1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&Z2_h, static_cast<size_t>(max_B) * out);
      alloc_h(&dZ2_h, static_cast<size_t>(max_B) * out);
      alloc_h(&dA1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&dZ1_h, static_cast<size_t>(max_B) * hid);
      alloc_h(&dW2_h, static_cast<size_t>(hid) * out);
      alloc_h(&dW1_h, static_cast<size_t>(in) * hid);
    }
  }

  /// Xavier-uniform weight initialisation (FP32 master weights).
  void init_weights(unsigned seed) {
    std::mt19937 rng(seed);
    auto xavier = [&](float* d_buf, int fan_in, int fan_out) {
      int n = fan_in * fan_out;
      float limit = std::sqrt(6.0F / static_cast<float>(fan_in + fan_out));
      std::uniform_real_distribution<float> dist(-limit, limit);
      std::vector<float> host(static_cast<size_t>(n));
      for (auto& v : host) v = dist(rng);
      CUDA_CHECK(cudaMemcpy(d_buf, host.data(), static_cast<size_t>(n) * sizeof(float),
                            cudaMemcpyHostToDevice));
    };
    xavier(W1, in_dim, hid_dim);
    xavier(W2, hid_dim, out_dim);
    CUDA_CHECK(cudaMemset(b1, 0, static_cast<size_t>(hid_dim) * sizeof(float)));
    CUDA_CHECK(cudaMemset(b2, 0, static_cast<size_t>(out_dim) * sizeof(float)));
  }

  // ------------------------------------------------------------------
  /// @name Forward pass
  /// @{
  // ------------------------------------------------------------------

  /// FP32 / TF32 forward pass.
  ///
  /// TF32 and FP32 use the **exact same code path** — the only difference
  /// is the math-mode flag set on the cuBLAS handle during `alloc()`.
  /// cuBLAS internally decides whether to use Tensor Cores.
  float forward_fp32(const float* d_X, const float* d_target, int B) {
    int blk = 256;

    // Z1[B×hid] = X[B×in] · W1[in×hid]
    gemm_fp32(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);

    // A1 = relu(Z1)
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);

    // Z2[B×out] = A1[B×hid] · W2[hid×out]
    gemm_fp32(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);

    // Log-softmax + cross-entropy loss.
    batch_log_softmax<<<B, 1>>>(Z2, LS, out_dim);

    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    ce_loss_kernel<<<1, 1>>>(LS, d_target, d_loss, B, out_dim);

    float h_loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss;
  }

  /// FP16 forward pass.
  ///
  /// 1. Convert FP32 master weights → FP16 shadow weights.
  /// 2. Convert FP32 input batch → FP16.
  /// 3. Run GEMMs in FP16 (Tensor Cores, FP32 accumulation).
  /// 4. Convert FP16 activations back to FP32 for the loss computation
  ///    (softmax in FP16 is numerically risky).
  float forward_fp16(const float* d_X, const float* d_target, int B) {
    int blk = 256;

    // Step 1: snapshot master weights into FP16 shadows.
    fp32_to_fp16(W1, W1_h, in_dim * hid_dim);
    fp32_to_fp16(W2, W2_h, hid_dim * out_dim);

    // Step 2: convert input to FP16.
    fp32_to_fp16(d_X, X_h, B * in_dim);

    // Z1_h[B×hid] = X_h[B×in] · W1_h[in×hid]   (FP16 GEMM)
    gemm_fp16(cublas, B, in_dim, hid_dim, 1.0F, X_h, W1_h, 0.0F, Z1_h);

    // Convert Z1_h → Z1 (FP32) for bias add and ReLU.
    fp16_to_fp32(Z1_h, Z1, B * hid_dim);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);

    // A1 = relu(Z1)
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);

    // Convert A1 → A1_h (FP16) for layer 2 GEMM.
    fp32_to_fp16(A1, A1_h, B * hid_dim);

    // Z2_h[B×out] = A1_h[B×hid] · W2_h[hid×out]  (FP16 GEMM)
    gemm_fp16(cublas, B, hid_dim, out_dim, 1.0F, A1_h, W2_h, 0.0F, Z2_h);

    // Convert Z2_h → Z2 (FP32) for bias + softmax.
    fp16_to_fp32(Z2_h, Z2, B * out_dim);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);

    // Log-softmax + CE loss — always in FP32 for accuracy.
    batch_log_softmax<<<B, 1>>>(Z2, LS, out_dim);

    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    ce_loss_kernel<<<1, 1>>>(LS, d_target, d_loss, B, out_dim);

    float h_loss = 0.0F;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss;
  }

  /// Dispatch forward pass based on precision mode.
  float forward(const float* d_X, const float* d_target, int B) {
    if (mode == PrecisionMode::kFP16) {
      return forward_fp16(d_X, d_target, B);
    }
    return forward_fp32(d_X, d_target, B);  // FP32 or TF32.
  }

  /// @}

  // ------------------------------------------------------------------
  /// @name Backward pass
  /// @{
  // ------------------------------------------------------------------

  /// FP32 / TF32 backward pass.
  ///
  /// Gradients for both TF32 and FP32 are computed identically.  TF32
  /// only affects the GEMM precision, not the gradient logic.
  void backward_fp32(const float* d_X, const float* d_target, int B) {
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);

    // dZ2 = softmax(Z2) − target  [B×out]
    ce_bwd<<<(B * out_dim + blk - 1) / blk, blk>>>(LS, d_target, dZ2, B * out_dim);

    // dW2 = (1/B) · A1^T · dZ2  [hid×out]
    gemm_fp32_AT(cublas, hid_dim, B, out_dim, inv_B, A1, dZ2, 0.0F, dW2);

    // db2 = (1/B) · col_sum(dZ2)
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);

    // dA1 = dZ2 · W2^T  [B×hid]
    {
      float one = 1.0F, zero = 0.0F;
      // W2 rm[hid×out] = cm[out×hid].  Need cm[hid×out] → OP_T.
      // dZ2 rm[B×out] = cm[out×B].  Need cm[out×B] → OP_N.
      // Result: cm[hid×B] = rm[B×hid].
      CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2,
                               out_dim, dZ2, out_dim, &zero, dA1, hid_dim));
    }

    // dZ1 = dA1 ⊙ relu'(Z1)
    relu_bwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, B * hid_dim);

    // dW1 = (1/B) · X^T · dZ1  [in×hid]
    gemm_fp32_AT(cublas, in_dim, B, hid_dim, inv_B, d_X, dZ1, 0.0F, dW1);

    // db1 = (1/B) · col_sum(dZ1)
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  /// FP16 backward pass with **loss scaling**.
  ///
  /// The overall flow:
  ///   1. Compute dZ2 in FP32 (from softmax output).
  ///   2. Apply loss scale: dZ2 *= loss_scale.
  ///   3. Convert dZ2 → FP16 for the GEMM operations.
  ///   4. Compute dW2, dA1 using FP16 GEMMs (Tensor Cores).
  ///   5. Convert results back to FP32.
  ///   6. Un-scale: dW2 /= loss_scale, dW1 /= loss_scale.
  ///   7. SGD update uses the un-scaled FP32 gradients.
  void backward_fp16(const float* d_X, const float* d_target, int B) {
    (void)d_X;  // FP16 path uses X_h (converted during forward).
    int blk = 256;
    float inv_B = 1.0F / static_cast<float>(B);

    // --- dZ2 = softmax(Z2) − target,  then scale ---
    ce_bwd<<<(B * out_dim + blk - 1) / blk, blk>>>(LS, d_target, dZ2, B * out_dim);
    scale_kernel<<<(B * out_dim + blk - 1) / blk, blk>>>(dZ2, loss_scale, B * out_dim);

    // Convert scaled dZ2 → FP16.
    fp32_to_fp16(dZ2, dZ2_h, B * out_dim);

    // --- dW2 = (1/B) · A1_h^T · dZ2_h  [hid×out] → FP16 ---
    // A1_h is the FP16 activation saved during forward.
    gemm_fp16_AT(cublas, hid_dim, B, out_dim, inv_B, A1_h, dZ2_h, 0.0F, dW2_h);
    fp16_to_fp32(dW2_h, dW2, hid_dim * out_dim);

    // --- db2 = (1/B) · col_sum(dZ2)  (already in FP32, but scaled) ---
    col_sum<<<(out_dim + blk - 1) / blk, blk>>>(dZ2, db2, B, out_dim);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_B, out_dim);

    // --- dA1 = dZ2_h · W2_h^T  [B×hid] → FP16 ---
    {
      float one = 1.0F, zero = 0.0F;
      // W2_h cm[out×hid] → need cm[hid×out] → OP_T.
      // dZ2_h cm[out×B] → OP_N.
      // Result: cm[hid×B] = rm[B×hid].
      CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, hid_dim, B, out_dim, &one, W2_h,
                                CUDA_R_16F, out_dim, dZ2_h, CUDA_R_16F, out_dim, &zero, dA1_h,
                                CUDA_R_16F, hid_dim, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    fp16_to_fp32(dA1_h, dA1, B * hid_dim);

    // --- dZ1 = dA1 ⊙ relu'(Z1)  (FP32) ---
    relu_bwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, dA1, dZ1, B * hid_dim);
    fp32_to_fp16(dZ1, dZ1_h, B * hid_dim);

    // --- dW1 = (1/B) · X_h^T · dZ1_h  [in×hid] → FP16 ---
    gemm_fp16_AT(cublas, in_dim, B, hid_dim, inv_B, X_h, dZ1_h, 0.0F, dW1_h);
    fp16_to_fp32(dW1_h, dW1, in_dim * hid_dim);

    // --- db1 = (1/B) · col_sum(dZ1)  (FP32, scaled) ---
    col_sum<<<(hid_dim + blk - 1) / blk, blk>>>(dZ1, db1, B, hid_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_B, hid_dim);

    // --- Un-scale all gradients ---
    float inv_scale = 1.0F / loss_scale;
    int n_dW2 = hid_dim * out_dim;
    int n_dW1 = in_dim * hid_dim;
    scale_kernel<<<(n_dW2 + blk - 1) / blk, blk>>>(dW2, inv_scale, n_dW2);
    scale_kernel<<<(n_dW1 + blk - 1) / blk, blk>>>(dW1, inv_scale, n_dW1);
    scale_kernel<<<(out_dim + blk - 1) / blk, blk>>>(db2, inv_scale, out_dim);
    scale_kernel<<<(hid_dim + blk - 1) / blk, blk>>>(db1, inv_scale, hid_dim);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  /// Dispatch backward pass based on precision mode.
  void backward(const float* d_X, const float* d_target, int B) {
    if (mode == PrecisionMode::kFP16) {
      backward_fp16(d_X, d_target, B);
    } else {
      backward_fp32(d_X, d_target, B);
    }
  }

  /// @}

  // ------------------------------------------------------------------
  /// @name SGD update
  /// @{
  // ------------------------------------------------------------------

  /// Apply SGD to all FP32 master weights.
  ///
  /// In FP16 mode, we update the FP32 master weights — the FP16 shadows
  /// are re-derived at the start of each forward pass.
  void sgd_step(float lr) {
    int blk = 256;
    int n1 = in_dim * hid_dim, n2 = hid_dim * out_dim;
    sgd_update<<<(n1 + blk - 1) / blk, blk>>>(W1, dW1, lr, n1);
    sgd_update<<<(hid_dim + blk - 1) / blk, blk>>>(b1, db1, lr, hid_dim);
    sgd_update<<<(n2 + blk - 1) / blk, blk>>>(W2, dW2, lr, n2);
    sgd_update<<<(out_dim + blk - 1) / blk, blk>>>(b2, db2, lr, out_dim);
  }

  /// @}

  // ------------------------------------------------------------------
  /// @name Prediction
  /// @{
  // ------------------------------------------------------------------

  /// Run inference and return predicted class per sample.
  ///
  /// Always uses FP32 path for simplicity (the master weights are FP32).
  std::vector<int> predict_batch(const float* d_X, int B) {
    int blk = 256;

    // Forward through both layers (FP32).
    gemm_fp32(cublas, B, in_dim, hid_dim, 1.0F, d_X, W1, 0.0F, Z1);
    add_bias<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, b1, B, hid_dim);
    relu_fwd<<<(B * hid_dim + blk - 1) / blk, blk>>>(Z1, A1, B * hid_dim);
    gemm_fp32(cublas, B, hid_dim, out_dim, 1.0F, A1, W2, 0.0F, Z2);
    add_bias<<<(B * out_dim + blk - 1) / blk, blk>>>(Z2, b2, B, out_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy logits to host and argmax.
    std::vector<float> logits(static_cast<size_t>(B) * out_dim);
    CUDA_CHECK(cudaMemcpy(logits.data(), Z2, static_cast<size_t>(B) * out_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<int> preds(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
      int best = 0;
      float best_val = logits[static_cast<size_t>(b) * out_dim];
      for (int c = 1; c < out_dim; ++c) {
        float v = logits[static_cast<size_t>(b) * out_dim + c];
        if (v > best_val) {
          best_val = v;
          best = c;
        }
      }
      preds[static_cast<size_t>(b)] = best;
    }
    return preds;
  }

  /// @}

  /// Free all device memory.
  void free_all() {
    auto safe_free = [](void* p) {
      if (p) CUDA_CHECK(cudaFree(p));
    };
    safe_free(W1);
    safe_free(b1);
    safe_free(W2);
    safe_free(b2);
    safe_free(Z1);
    safe_free(A1);
    safe_free(Z2);
    safe_free(LS);
    safe_free(dZ2);
    safe_free(dA1);
    safe_free(dZ1);
    safe_free(dW1);
    safe_free(db1);
    safe_free(dW2);
    safe_free(db2);
    safe_free(W1_h);
    safe_free(W2_h);
    safe_free(X_h);
    safe_free(Z1_h);
    safe_free(A1_h);
    safe_free(Z2_h);
    safe_free(dZ2_h);
    safe_free(dA1_h);
    safe_free(dZ1_h);
    safe_free(dW2_h);
    safe_free(dW1_h);
    if (cublas) CUBLAS_CHECK(cublasDestroy(cublas));
    cublas = nullptr;
  }
};

// =============================================================================
// Data generation (same synthetic 3-class problem as Lessons 17 & 20)
// =============================================================================

/// Generate 2-D Gaussian clusters projected to 4-D features.
void generate_data(std::vector<float>& flat, std::vector<int>& labels, int n_per_class,
                   unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0F, 0.3F);
  // 3 cluster centres in 2-D, repeated to fill 4 features.
  const float centres[3][4] = {
      {0.0F, 0.0F, 0.0F, 0.0F},
      {3.0F, 3.0F, 3.0F, 3.0F},
      {-3.0F, 3.0F, -3.0F, 3.0F},
  };
  flat.clear();
  labels.clear();
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      for (int f = 0; f < 4; ++f) {
        flat.push_back(centres[c][f] + noise(rng));
      }
      labels.push_back(c);
    }
  }
}

/// Build one-hot matrix [N × C] from integer labels.
std::vector<float> make_one_hot(const std::vector<int>& labels, int C) {
  size_t N = labels.size();
  std::vector<float> oh(N * static_cast<size_t>(C), 0.0F);
  for (size_t i = 0; i < N; ++i) {
    oh[i * static_cast<size_t>(C) + static_cast<size_t>(labels[i])] = 1.0F;
  }
  return oh;
}

// =============================================================================
// Training helper
// =============================================================================

/// Train the MLP for a fixed number of epochs, return final accuracy.
static float train_and_evaluate(PrecisionMode mode, const char* label) {
  constexpr int N_PER_CLASS = 50;
  constexpr int N_TOTAL = 3 * N_PER_CLASS;
  constexpr int BATCH_SIZE = 32;
  constexpr int EPOCHS = 50;
  constexpr float LR = 0.1F;

  std::vector<float> flat_data;
  std::vector<int> labels;
  generate_data(flat_data, labels, N_PER_CLASS, 42);
  std::vector<float> one_hot = make_one_hot(labels, 3);

  MixedPrecisionMLP mlp{};
  mlp.alloc(4, 16, 3, BATCH_SIZE, mode);
  mlp.init_weights(123);

  float *d_bx, *d_bt;
  CUDA_CHECK(cudaMalloc(&d_bx, static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bt, static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float)));

  std::vector<int> indices(N_TOTAL);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(7);

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
      std::vector<float> bx(static_cast<size_t>(BATCH_SIZE) * 4);
      std::vector<float> bt(static_cast<size_t>(BATCH_SIZE) * 3);
      for (int b = 0; b < BATCH_SIZE; ++b) {
        int si = indices[static_cast<size_t>(start + b)];
        std::memcpy(&bx[static_cast<size_t>(b) * 4], &flat_data[static_cast<size_t>(si) * 4],
                    4 * sizeof(float));
        std::memcpy(&bt[static_cast<size_t>(b) * 3], &one_hot[static_cast<size_t>(si) * 3],
                    3 * sizeof(float));
      }
      CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), static_cast<size_t>(BATCH_SIZE) * 3 * sizeof(float),
                            cudaMemcpyHostToDevice));

      mlp.forward(d_bx, d_bt, BATCH_SIZE);
      mlp.backward(d_bx, d_bt, BATCH_SIZE);
      mlp.sgd_step(LR);
    }
  }

  // Evaluate on training set in chunks of BATCH_SIZE to respect buffer sizes.
  int correct = 0;
  for (int start = 0; start + BATCH_SIZE <= N_TOTAL; start += BATCH_SIZE) {
    CUDA_CHECK(cudaMemcpy(d_bx, &flat_data[static_cast<size_t>(start) * 4],
                          static_cast<size_t>(BATCH_SIZE) * 4 * sizeof(float),
                          cudaMemcpyHostToDevice));
    auto preds = mlp.predict_batch(d_bx, BATCH_SIZE);
    for (int b = 0; b < BATCH_SIZE; ++b) {
      if (preds[static_cast<size_t>(b)] == labels[static_cast<size_t>(start + b)]) {
        ++correct;
      }
    }
  }
  int evaluated = (N_TOTAL / BATCH_SIZE) * BATCH_SIZE;
  float accuracy = static_cast<float>(correct) / static_cast<float>(evaluated) * 100.0F;

  std::printf("[%s] Accuracy: %.1f%% (%d/%d)\n", label, accuracy, correct, evaluated);

  mlp.free_all();
  CUDA_CHECK(cudaFree(d_bx));
  CUDA_CHECK(cudaFree(d_bt));

  return accuracy;
}

// =============================================================================
// main — compare all three precision modes
// =============================================================================

int main() {
  std::printf("=== Lesson 21: Mixed-Precision Training ===\n\n");

  // Check GPU compute capability.
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

  if (prop.major < 7) {
    std::printf(
        "WARNING: Tensor Cores require sm_70+ (Volta).  "
        "FP16 GEMM will fall back to CUDA cores.\n\n");
  }
  if (prop.major < 8) {
    std::printf(
        "NOTE: TF32 requires sm_80+ (Ampere).  "
        "TF32 mode will behave like FP32 on this GPU.\n\n");
  }

  float acc_fp32 = train_and_evaluate(PrecisionMode::kFP32, "FP32");
  float acc_tf32 = train_and_evaluate(PrecisionMode::kTF32, "TF32 ");
  float acc_fp16 = train_and_evaluate(PrecisionMode::kFP16, "FP16");

  std::printf("\n--- Summary ---\n");
  std::printf("FP32 accuracy:  %.1f%%\n", acc_fp32);
  std::printf("TF32 accuracy:  %.1f%%\n", acc_tf32);
  std::printf("FP16 accuracy:  %.1f%%\n", acc_fp16);
  std::printf("\nAll three modes should achieve similar accuracy (>90%%).\n");
  std::printf("On Ampere+ GPUs, TF32/FP16 GEMMs use Tensor Cores for\n");
  std::printf("significant speed-ups on larger matrices.\n");

  return 0;
}
