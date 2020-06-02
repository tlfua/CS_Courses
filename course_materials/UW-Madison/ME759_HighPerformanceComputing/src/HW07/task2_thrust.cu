#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);

  thrust::host_vector<float> h_vec(n, 1);
  thrust::device_vector<float> d_vec_in(n);
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec_in.begin());
  thrust::device_vector<float> d_vec_out(n);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  cudaEventRecord(start);
  thrust::exclusive_scan(d_vec_in.begin(), d_vec_in.end(), d_vec_out.begin());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << d_vec_out[n - 1] << "\n";
  std::cout << ms << "\n";

  return 0;
}
