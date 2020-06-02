#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);

  thrust::host_vector<int> h_vec(n, 1);
  thrust::device_vector<int> d_vec(n);
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  cudaEventRecord(start);
  int res = thrust::reduce(d_vec.begin(), d_vec.end());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << res << "\n";
  std::cout << ms << "\n";

  return 0;
}