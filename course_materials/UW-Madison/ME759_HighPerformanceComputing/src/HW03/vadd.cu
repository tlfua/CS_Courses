#include "vadd.cuh"

__global__ void vadd(const float *a, float *b, unsigned int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    b[index] += a[index];
  }
}