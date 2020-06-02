// task1.cpp : This file contains the 'main' function. Program execution begins
// and ends there.

// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include "omp.h"
#include "optimize.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
//#define debug

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void test(vec *v, data_t *dest, void (*optimize)(vec *, data_t *));

int main(int argc, char *argv[]) {

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  int n = 8;

  if (argc < 1) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  n = std::stoi(argv[1], nullptr, 0);

  vec *my_vec = new vec(n);
  my_vec->data = new (std::nothrow) data_t[n];

  // another way
  // vec c(n);
  // c.data = new (std::nothrow) data_t[n];

  if (!my_vec->data) {
    std::cout << "Memory allocation failed\n";
    return 0;
  }

  data_t *dest = new (std::nothrow) data_t[1];

  // Creates an array of n random double
  for (int i = 0; i < n; i++) {
    // my_vec->data[i] = std::rand() % (n+1);
    my_vec->data[i] = 1;
    // c.data[i]=1;
  }

#ifdef debug
  std::cout << "data:";
  for (int i = 0; i < n; i++)
    std::cout << my_vec->data[i] << " ";
  std::cout << std::endl;

#endif

  test(my_vec, dest, &optimize1);
  test(my_vec, dest, &optimize2);
  test(my_vec, dest, &optimize3);
  test(my_vec, dest, &optimize4);
  test(my_vec, dest, &optimize5);

  delete[] my_vec->data;
  delete[] dest;

  return 0;
}

void test(vec *v, data_t *dest, void (*optimize)(vec *, data_t *)) {

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  double total = 0;

  for (int loop = 0; loop < 100; loop++) {

    *dest = 0;
    // Get the starting timestamp
    start = high_resolution_clock::now();

    optimize(v, dest);

    // Get the ending timestamp
    end = high_resolution_clock::now();

    // Convert the calculated duration to a double using the standard library
    duration_sec =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    total += duration_sec.count();
  }

  std::cout << *dest << std::endl;
  std::cout << total/100 << "\n";
}
