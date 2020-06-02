#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>
#include <iostream>
#include <new>
#include <stdio.h>
// #include "../../test/test_util.h"
using namespace cub;

// Globals, constants and typedefs
// bool                    g_verbose = false;  // Whether to display
// input/output to console
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

void fill_one(float *h_in, size_t num_items) {
  for (unsigned int i = 0; i < num_items; i++) {
    h_in[i] = 1;
  }
}

int main(int argc, char **argv) {
  size_t num_items = atoi(argv[1]);

  // Allocate host arrays
  float *h_in = new (std::nothrow) float[num_items];

  // Initialize problem and solution
  fill_one(h_in, num_items);

  // Allocate problem device arrays
  float *d_in = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_in, sizeof(float) * num_items));

  // Initialize device input
  CubDebugExit(
      cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

  // Allocate device output array
  float *d_out = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_out, sizeof(float) * num_items));

  // Allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        d_in, d_out, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  // Run
  cudaEventRecord(start);
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        d_in, d_out, num_items));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  float *h_out = new (std::nothrow) float[num_items];
  CubDebugExit(cudaMemcpy(h_out, d_out, sizeof(float) * num_items,
                          cudaMemcpyDeviceToHost));

  std::cout << h_out[num_items - 1] << "\n";
  std::cout << ms << "\n";

  // Cleanup
  if (h_in)
    delete[] h_in;
  if (h_out)
    delete[] h_out;
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out)
    CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  return 0;
}
