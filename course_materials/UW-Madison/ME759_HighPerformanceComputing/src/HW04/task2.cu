#include "stencil.cuh"
#include <cstdio>
#include <cstdlib>
#include <iostream>

void test1(float *image, float *mask) {
  image[0] = 4.0;
  image[1] = 1.0;
  image[2] = 2.0;
  image[3] = 3.0;

  mask[0] = 1.0;
  mask[1] = 2.0;
  mask[2] = 1.0;
}

void test2(float *image, float *mask) {
  image[0] = 1.0;
  image[1] = 0.0;
  image[2] = 0.0;
  image[3] = 0.0;
  image[4] = 1.0;
  image[5] = 0.0;
  image[6] = 0.0;
  image[7] = 0.0;
  image[8] = 1.0;
  image[9] = 0.0;

  mask[0] = 1.0;
  mask[1] = 2.0;
  mask[2] = 1.0;
}

void assign_input(float *image, float *mask, unsigned int n, unsigned int R) {
  for (unsigned int i = 0; i < n; ++i) {
    image[i] = 1.0;
  }

  for (unsigned int i = 0; i < 2 * R + 1; ++i) {
    mask[i] = 1.0;
  }
}

int main(int argc, char *argv[]) {
  unsigned int n = atoi(argv[1]);
  unsigned int R = atoi(argv[2]);
  unsigned int threads_per_block = atoi(argv[3]);

  float *image, *mask, *output;
  cudaMallocManaged(&image, n * sizeof(float));
  cudaMallocManaged(&mask, (2 * R + 1) * sizeof(float));
  cudaMallocManaged(&output, n * sizeof(float));

  // sample input
  if ((n == 4) && (R == 1)) {
    test1(image, mask);
  } else if ((n == 10) && (R == 1)) {
    test2(image, mask);
  } else { // general input
    assign_input(image, mask, n, R);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  stencil(image, mask, output, n, R, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // sample print
  /*
  for (unsigned int i = 0; i < n; ++i) {
      std::cout << output[i] << " ";
  }
  */

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << output[n - 1] << '\n';
  std::cout << ms << '\n';

  cudaFree(image);
  cudaFree(mask);
  cudaFree(output);
  return 0;
}
