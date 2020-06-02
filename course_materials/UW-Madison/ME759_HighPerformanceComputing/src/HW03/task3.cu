#include "vadd.cuh"
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define THREAD_NUM 512
// #define THREAD_NUM 1024

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  float *a, *b;
  cudaMallocManaged(&a, N * sizeof(float));
  cudaMallocManaged(&b, N * sizeof(float));

  // assign a and b
  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = (float)i;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  vadd<<<N / THREAD_NUM + 1, THREAD_NUM>>>(a, b, N);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // for (int i = 0; i < N; i++) {
  //     std::cout << b[i] << " ";
  // }

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << ms / 1000 << "\n";
  std::cout << b[0] << "\n";
  std::cout << b[N - 1] << "\n";

  cudaFree(a);
  cudaFree(b);
  return 0;
}
