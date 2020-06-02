#include <iostream>
// #include <cstdio>
// #include <cstdlib>

#define BLOCK_NUM 2
#define THREAD_NUM 8
#define N 16

__global__ void addBlockIdxAndThreadIdx(int *output) {
  output[blockDim.x * blockIdx.x + threadIdx.x] = blockIdx.x + threadIdx.x;
}

int main() {
  int *d_arr;
  cudaMallocManaged(&d_arr, N * sizeof(int));

  addBlockIdxAndThreadIdx<<<BLOCK_NUM, THREAD_NUM>>>(d_arr);
  cudaDeviceSynchronize();

  int h_arr[N];
  cudaMemcpy(&h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    // h_arr[i] = d_arr[i];

    std::cout << h_arr[i];
    if (i < N - 1) {
      std::cout << " ";
    } else {
      std::cout << "\n";
    }
  }

  cudaFree(d_arr);
  return 0;
}