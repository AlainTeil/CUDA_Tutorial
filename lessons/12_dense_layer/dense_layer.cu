/**
 * @file dense_layer.cu
 * @brief Lesson 12 — Dense (Fully Connected) Layer: forward and backward.
 *
 * A dense (fully connected) layer is the simplest neural-network building
 * block.  Every input feature is connected to every output neuron:
 *
 * ```
 * Forward:   Y  = X · W  + b
 * Backward:  dX = dY · Wᵀ       (“which inputs caused this error?”)
 *            dW = Xᵀ · dY       (“how should the weights change?”)
 *            db = sum(dY, axis=0) (“how should the biases change?”)
 * ```
 *
 * Where:
 *   - X  : (batch × in_dim)   — input data (one row per sample)
 *   - W  : (in_dim × out_dim) — learnable weight matrix
 *   - b  : (1 × out_dim)      — learnable bias vector (broadcast over batch)
 *   - Y  : (batch × out_dim)  — output
 *   - dY : (batch × out_dim)  — upstream gradient (from the loss or next layer)
 *
 * ## Implementation notes
 *
 * - The forward pass is one matmul + a bias add kernel.
 * - The backward pass requires **two transposes** (Wᵀ, Xᵀ) and two matmuls.
 *   We reuse the tiled matmul from Lesson 11.  In later lessons (19–21),
 *   we switch to cuBLAS for higher performance.
 * - The bias gradient is a column-wise sum of dY — one thread per output
 *   column iterates over the batch dimension.  For large batches, a
 *   parallel reduction per column would be faster.
 *
 * ## Gradient checking
 *
 * The test file verifies correctness with **numerical gradient checking**:
 * perturb each parameter by ε, re-run the forward pass, and compare the
 * finite-difference gradient to the analytical backward pass.  This is the
 * gold standard for catching gradient bugs.
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

constexpr int kTile = 16;

// =============================================================================
// Tiled matmul (reused from Lesson 11)
// =============================================================================

/// @brief Tiled GEMM kernel: C = A(M×K) × B(K×N).
///
/// Identical to the kernel in Lesson 11.  It is duplicated here because each
/// lesson compiles independently (no shared library).  Later lessons (19–21)
/// replace this with `cublasSgemm` for production-quality performance.
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];
  int col = blockIdx.x * kTile + threadIdx.x;
  int row = blockIdx.y * kTile + threadIdx.y;
  float sum = 0.0F;
  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    int a_col = t * kTile + threadIdx.x;
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0F;
    int b_row = t * kTile + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0F;
    __syncthreads();
    for (int k = 0; k < kTile; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = sum;
}

// =============================================================================
// Transpose kernel (for computing Wᵀ and Xᵀ)
// =============================================================================

/// Naive transpose: `out[c][r] = in[r][c]`.
/// Used to prepare Wᵀ for `dX = dY · Wᵀ` and Xᵀ for `dW = Xᵀ · dY`.
/// For larger matrices, the tiled transpose from Lesson 10 would be faster.
__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    out[col * rows + row] = in[row * cols + col];
  }
}

// =============================================================================
// Bias add: Y[i][j] += b[j]
// =============================================================================

/// @brief Add bias vector to every row of Y.
///
/// Each thread handles one (row, col) element.  The bias is **broadcast**
/// over the batch dimension — the same `b[col]` is added to every row.
/// This is logically equivalent to `Y += ones(batch,1) · b`.
__global__ void add_bias(float* Y, const float* b, int batch, int out_dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch && col < out_dim) {
    Y[row * out_dim + col] += b[col];
  }
}

// =============================================================================
// Bias gradient: db[j] = sum_i dY[i][j]
// =============================================================================

/// @brief Compute bias gradient by summing dY along the batch axis.
///
/// Each thread handles one output column and loops over all batch rows.
/// This is O(batch) work per thread.  For large batches, a parallel
/// reduction (one block per column, tree reduction along rows) would be
/// more efficient.  For the small batches used in this tutorial the serial
/// loop is sufficient.
__global__ void bias_grad(const float* dY, float* db, int batch, int out_dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < out_dim) {
    float sum = 0.0F;
    for (int i = 0; i < batch; ++i) {
      sum += dY[i * out_dim + col];
    }
    db[col] = sum;
  }
}

// =============================================================================
// Host helpers
// =============================================================================

/// Launch matmul: C = A(M×K) * B(K×N).
/// Grid dimensions: ceil(N/TILE) blocks in x, ceil(M/TILE) in y.
void gpu_matmul(const float* dA, const float* dB, float* dC, int M, int N, int K) {
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  matmul_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
}

/// Launch transpose: out = in^T  (in is rows×cols, out is cols×rows).
void gpu_transpose(const float* d_in, float* d_out, int rows, int cols) {
  dim3 threads(kTile, kTile);
  dim3 blocks((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_kernel<<<blocks, threads>>>(d_in, d_out, rows, cols);
}

// =============================================================================
// Dense layer forward: Y = X * W + b
// =============================================================================

/// @brief Compute forward pass of a fully-connected layer.
///
/// Steps:
///   1. GEMM: Y = X(batch×in_dim) × W(in_dim×out_dim)
///   2. Bias: Y[i][j] += b[j]  for all i
///
/// After this call, `dY` holds the layer output in row-major order.
void dense_forward(const float* dX, const float* dW, const float* db, float* dY, int batch,
                   int in_dim, int out_dim) {
  gpu_matmul(dX, dW, dY, batch, out_dim, in_dim);

  dim3 threads(kTile, kTile);
  dim3 blocks((out_dim + kTile - 1) / kTile, (batch + kTile - 1) / kTile);
  add_bias<<<blocks, threads>>>(dY, db, batch, out_dim);
}

// =============================================================================
// Dense layer backward
// =============================================================================

/// @brief Compute backward pass of a fully-connected layer.
///
/// Given upstream gradient `dY_grad` (batch×out_dim), compute:
///   - `dX_grad = dY_grad · Wᵀ`  (batch×in_dim) — gradient w.r.t. input.
///   - `dW_grad = Xᵀ · dY_grad` (in_dim×out_dim) — gradient w.r.t. weights.
///   - `db_grad = sum(dY_grad, axis=0)` (1×out_dim) — gradient w.r.t. bias.
///
/// The transposes allocate temporary device memory.  For a production
/// training loop, these buffers would be pre-allocated to avoid the
/// overhead of repeated `cudaMalloc` / `cudaFree`.
void dense_backward(const float* dX, const float* dW, const float* dY_grad, float* dX_grad,
                    float* dW_grad, float* db_grad, int batch, int in_dim, int out_dim) {
  // Temporary buffers for transposed matrices
  float* dWt = nullptr;
  float* dXt = nullptr;
  CUDA_CHECK(cudaMalloc(&dWt, static_cast<size_t>(out_dim) * in_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dXt, static_cast<size_t>(in_dim) * batch * sizeof(float)));

  // dX_grad = dY_grad * W^T
  gpu_transpose(dW, dWt, in_dim, out_dim);
  gpu_matmul(dY_grad, dWt, dX_grad, batch, in_dim, out_dim);

  // dW_grad = X^T * dY_grad
  gpu_transpose(dX, dXt, batch, in_dim);
  gpu_matmul(dXt, dY_grad, dW_grad, in_dim, out_dim, batch);

  // db_grad = sum(dY_grad, axis=0)
  bias_grad<<<(out_dim + 255) / 256, 256>>>(dY_grad, db_grad, batch, out_dim);

  CUDA_CHECK(cudaFree(dWt));
  CUDA_CHECK(cudaFree(dXt));
}

// =============================================================================
// main — quick smoke test
// =============================================================================
int main() {
  constexpr int kBatch = 4, kIn = 8, kOut = 3;

  std::vector<float> hX(kBatch * kIn), hW(kIn * kOut), hb(kOut);
  srand(42);
  for (auto& v : hX) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hW) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;
  for (auto& v : hb) v = static_cast<float>(rand() / static_cast<double>(RAND_MAX)) - 0.5F;

  float *dX, *dW, *db, *dY;
  CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, hb.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, static_cast<size_t>(kBatch) * kOut * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * sizeof(float), cudaMemcpyHostToDevice));

  dense_forward(dX, dW, db, dY, kBatch, kIn, kOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hY(kBatch * kOut);
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("Dense forward output (first row):\n");
  for (int j = 0; j < kOut; ++j)
    std::printf("  Y[0][%d] = %.4f\n", j, static_cast<double>(hY[static_cast<size_t>(j)]));

  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dY));
  return EXIT_SUCCESS;
}
