# Lesson 19 — cuBLAS & cuDNN

## Objective

Replace hand-written matmul and convolution kernels with vendor-optimised
cuBLAS and cuDNN library calls for production-grade performance.

## Prerequisites

- Lesson 11 (Matmul), Lesson 14 (Conv2D).

## Key Concepts

| Concept | Description |
|---------|-------------|
| cuBLAS SGEMM | Drop-in GEMM replacement; Tensor-Core-enabled |
| Row-major trick | C_rm = A_rm · B_rm ⟺ Cᵀ_cm = Bᵀ_cm · Aᵀ_cm |
| cuDNN conv | Descriptor-based API with algorithm auto-selection |
| Workspace | cuDNN algorithms may require scratch memory |

## Build & Run

```bash
# cuDNN is optional — enabled via CMake flag:
cmake -B build -DCUDA_TUTORIAL_USE_CUDNN=ON
cmake --build build --target 19_cublas_cudnn
./build/lessons/19_cublas_cudnn/19_cublas_cudnn
```

## Run Tests

```bash
ctest --test-dir build -R 19_cublas_cudnn
```

## What You'll Learn

1. The elegant row-major ↔ column-major trick for cuBLAS.
2. How cuDNN's descriptor-based API differs from raw kernel calls.
3. Why understanding low-level kernels is a prerequisite for library use.
