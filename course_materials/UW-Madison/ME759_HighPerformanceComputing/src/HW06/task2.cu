#include "scan.cuh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>

#define THREADS_PER_BLOCK 1024

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  // int threads_per_block = atoi(argv[2]);

  float *in = new (std::nothrow) float[N];
  for (int i = 0; i < N; ++i) {
    in[i] = 1;
  }
  float *out = new (std::nothrow) float[N];

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  scan(in, out, N, THREADS_PER_BLOCK);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << out[N - 1] << '\n';
  std::cout << ms << '\n';

  delete[] in;
  delete[] out;
  return 0;
}
