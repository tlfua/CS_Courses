#include <cstdio>
#include <cstdlib>
#include <iostream>

#define BLOCK_NUM 1
#define THREAD_NUM 4

__global__ void print() {
  std::printf("Hello World! I am thread %d\n", threadIdx.x);
}

int main() {
  print<<<BLOCK_NUM, THREAD_NUM>>>();
  cudaDeviceSynchronize();

  return 0;
}