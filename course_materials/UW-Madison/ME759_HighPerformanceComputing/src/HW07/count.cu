

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "count.cuh"

void count(const thrust::device_vector<int> &d_in,
           thrust::device_vector<int> &out_values,
           thrust::device_vector<int> &out_counts) {
  thrust::device_vector<int> in_values(d_in.size());
  thrust::copy(d_in.begin(), d_in.end(), in_values.begin());

  thrust::sort(in_values.begin(), in_values.end());

  thrust::device_vector<int> in_counts(d_in.size(), 1);
  // out_values.resize(d_in.size());
  // out_counts.resize(d_in.size());

  thrust::pair<thrust::device_vector<int>::iterator,
               thrust::device_vector<int>::iterator>
      out_end_pair =
          thrust::reduce_by_key(in_values.begin(), in_values.end(),
                                in_counts.begin(), out_values.begin(),
                                out_counts.begin());

  int out_size = thrust::distance(out_values.begin(), out_end_pair.first);
  out_values.resize(out_size);
  out_counts.resize(out_size);
}
