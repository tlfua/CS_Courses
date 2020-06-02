#include <iostream>
#include <new>

#include <chrono>
#include <ratio>

#include "convolution.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void assign_input(float *input) {
  float sample_input[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
  for (int i = 0; i < 16; i++) {
    input[i] = sample_input[i];
  }
}

void assign_mask(float *mask) {
  float sample_mask[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
  for (int i = 0; i < 9; i++) {
    mask[i] = sample_mask[i];
  }
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);

  float *input = new (std::nothrow) float[n * n];
  float *output = new (std::nothrow) float[n * n];
  float *mask = new (std::nothrow) float[3 * 3];
  if ((!input) || (!output) || (!mask)) {
    std::cout << "Can not allocate memory for either input or output or mask\n";
    return -1;
  }

  assign_input(input);
  assign_mask(mask);

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  start = high_resolution_clock::now();
  Convolve(input, output, n, mask, 3);
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  std::cout << duration_sec.count() << "\n";
  std::cout << output[0] << "\n";
  std::cout << output[n * n - 1] << "\n";

  delete[] input;
  delete[] output;
  delete[] mask;

  return 0;
}