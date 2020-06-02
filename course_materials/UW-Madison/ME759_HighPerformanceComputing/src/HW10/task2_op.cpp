// task1.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//

// iostream is not needed for timers, but we need it for cout
#include "reduce.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
#include <omp.h>
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  float *arr;
  double sum;

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  int n = std::stoi(argv[1], nullptr, 0);
  int numThread = std::stoi(argv[2], nullptr, 0);


  arr = new (std::nothrow) float[n];

  if(!arr){
    std::cout << "Memory allocation failed\n";
    return 0;
  }

  for(int i=0; i<n ; i++)
      arr[i] = 1.0;

  sum = 0;
  omp_set_num_threads(numThread);

  double total = 0;
  for (int loop = 0; loop < 20; loop++) {
  // Get the starting timestamp
  start = high_resolution_clock::now();

  sum = reduce(arr, 0, n);

  // Get the ending timestamp
  end = high_resolution_clock::now();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

   total += duration_sec.count();
  }

  // Prints the ﬁrst element of the output scanned array
  std::cout << sum << std::endl;

  // Durations are converted to milliseconds already thanks to
  // std::chrono::duration_cast
  std::cout << total/20 << "\n";

  delete[] arr;

  return 0;
}
