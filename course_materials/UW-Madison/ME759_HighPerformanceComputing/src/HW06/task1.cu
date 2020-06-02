#include "mmul.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

void assign_one(float *mat, unsigned int n) {
  for (unsigned int i = 0; i < n * n; ++i) {
    mat[i] = 1;
  }
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int n_ntests = atoi(argv[2]);

  float *A, *B, *C;
  cudaMallocManaged(&A, n * n * sizeof(float));
  cudaMallocManaged(&B, n * n * sizeof(float));
  cudaMallocManaged(&C, n * n * sizeof(float));

  assign_one(A, n);
  assign_one(B, n);
  assign_one(C, n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;
  float ms_sum = 0;
  float ms_avg;

  for (unsigned int i = 0; i < n_ntests; ++i) {
    cudaEventRecord(start);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    mmul(handle, A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    ms_sum += ms;
  }
  ms_avg = (float)(ms_sum / n_ntests);

  // std::cout << C[n * n - 1] << '\n';
  std::cout << ms_avg << '\n';

  cublasDestroy(handle);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
