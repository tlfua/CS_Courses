

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "count.cuh"

void sample_fill(thrust::host_vector<int> &h_vec) {
  h_vec[0] = 3;
  h_vec[1] = 5;
  h_vec[2] = 1;
  h_vec[3] = 2;
  h_vec[4] = 3;
  h_vec[5] = 1;
}

void general_fill(thrust::host_vector<int> &h_vec) {
  for (long unsigned int i = 0; i < h_vec.size(); ++i) {
    if ((i >> 1 << 1) == i) {
      h_vec[i] = 2;
    } else {
      h_vec[i] = 1;
    }
  }
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);

  thrust::host_vector<int> h_vec(n);
  if (h_vec.size() == 6) {
    sample_fill(h_vec);
  } else {
    general_fill(h_vec);
  }

  thrust::device_vector<int> d_values_in(n);
  thrust::copy(h_vec.begin(), h_vec.end(), d_values_in.begin());
  thrust::device_vector<int> d_values_out(n);
  thrust::device_vector<int> d_counts_out(n);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  cudaEventRecord(start);
  count(d_values_in, d_values_out, d_counts_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  /*
  // sample print
  for (long unsigned int i = 0; i < d_values_out.size(); ++i) {
      std::cout << d_values_out[i] << ": " << d_counts_out[i] << ",  ";
  }
  std::cout << "\n";
  */

  std::cout << d_values_out[d_values_out.size() - 1] << "\n";
  std::cout << d_counts_out[d_counts_out.size() - 1] << "\n";
  std::cout << ms << "\n";

  return 0;
}
