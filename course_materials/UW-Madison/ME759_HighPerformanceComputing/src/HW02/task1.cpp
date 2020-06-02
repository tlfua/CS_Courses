#include <iostream>
// #include <cstdio>
// #include <cstdlib>
#include <chrono>
#include <new>
#include <ratio>

#include "scan.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);

  float *input = new (std::nothrow) float[n];
  float *output = new (std::nothrow) float[n];
  if ((!input) || (!output)) {
    std::cout << "Can not allocate memory for either input or output\n";
    return -1;
  }

  // assign input
  for (int i = 0; i < n; i++) {
    input[i] = (float)(i + 1) / (float)(n + 1);
  }

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  start = high_resolution_clock::now();
  Scan(input, output, n);
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  std::cout << duration_sec.count() << "\n";
  std::cout << output[0] << "\n";
  std::cout << output[n - 1] << "\n";

  delete[] input;
  delete[] output;

  return 0;
}