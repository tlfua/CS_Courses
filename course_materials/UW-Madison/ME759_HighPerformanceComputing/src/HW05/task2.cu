#include "matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {

  int n = atoi(argv[1]);
  int block_dim = atoi(argv[2]);

  float *A, *B, *C;
  cudaMallocManaged(&A, n * n * sizeof(float));
  cudaMallocManaged(&B, n * n * sizeof(float));
  cudaMallocManaged(&C, n * n * sizeof(float));

  // general input
  for (int i = 0; i < n * n; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul(A, B, C, n, block_dim);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // sample print
  // for (int i = 0; i < n * n; ++i) {
  //     std::cout << C[i] << " ";
  // }

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << C[0] << '\n';
  std::cout << C[n * n - 1] << '\n';
  std::cout << ms << '\n';

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
