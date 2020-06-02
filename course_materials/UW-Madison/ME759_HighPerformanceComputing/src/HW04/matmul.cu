#include "matmul.cuh"

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              size_t n) {
  int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_index >= n * n) {
    return;
  }

  // transform 2D to 1D
  int idx = thread_index / n;
  int idy = thread_index % n;

  float sum = 0;
  for (size_t k = 0; k < n; ++k) {
    sum += A[idx * n + k] * B[k * n + idy];
  }

  C[idx * n + idy] = sum;
}

void matmul(const float *A, const float *B, float *C, size_t n,
            unsigned int threads_per_block) {
  int blocks;
  if ((n * n) % threads_per_block == 0) {
    blocks = (n * n) / threads_per_block;
  } else {
    blocks = (n * n) / threads_per_block + 1;
  }
  matmul_kernel<<<blocks, threads_per_block>>>(A, B, C,
                                                                      n);
  cudaDeviceSynchronize();
}