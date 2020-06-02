#include "reduce.cuh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);

  int *arr = new (std::nothrow) int[N];
  for (int i = 0; i < N; ++i) {
    arr[i] = 1;
  }

  // memset(arr, 1, sizeof(arr));  WRONG !!
  // std::cout << "main arr[0] = " << arr[0] << '\n';

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int sum = reduce(arr, N, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << sum << '\n';
  std::cout << ms << '\n';

  delete[] arr;
  return 0;
}
