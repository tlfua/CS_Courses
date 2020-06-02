#include <iostream>
#include <new>
#include <vector>

#include <chrono>
#include <ratio>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void transpose(const double *A, double *A_T, const std::size_t n) {
  int len = (int)n * n;
  for (int i = 0; i < len; i++) {
    int row_iter = i / n;
    int col_iter = i % n;
    A_T[col_iter * n + row_iter] = A[i];
  }
}

void do_mmul_and_print(
    const std::vector<void (*)(const double *, const double *, double *,
                               const std::size_t)> &mmuls,
    double *A, double *A_T, double *B, double *B_T, double *C,
    const std::size_t n) {
  double *first_mat;
  double *second_mat;
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  for (int i = 0; i < 4; i++) {
    first_mat = A;
    second_mat = B;
    if (i == 2) {
      second_mat = B_T;
    }
    if (i == 3) {
      first_mat = A_T;
    }

    start = high_resolution_clock::now();
    mmuls[i](first_mat, second_mat, C, n);
    end = high_resolution_clock::now();

    duration_sec =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_sec.count() << "\n";

    std::cout << C[n * n - 1] << "\n";
  }
}

int main() {
  int n = 1000;
  std::cout << n << "\n";

  double *A = new (std::nothrow) double[n * n];
  double *B = new (std::nothrow) double[n * n];
  double *C = new (std::nothrow) double[n * n];
  double *A_T = new (std::nothrow) double[n * n];
  double *B_T = new (std::nothrow) double[n * n];
  if ((!A) || (!B) || (!C) || (!A_T) || (!B_T)) {
    std::cout
        << "Can not allocate memory for either A or B or C or A_T or B_T\n";
    return -1;
  }

  // assign values of A and B
  for (int i = 0; i < n * n; i++) {
    if (i % 2 == 0) {
      A[i] = 0;
      B[i] = 0;
    } else {
      A[i] = 1;
      B[i] = 1;
    }
  }

  transpose(A, A_T, n);
  transpose(B, B_T, n);

  std::vector<void (*)(const double *, const double *, double *,
                       const std::size_t)>
      mmuls;
  mmuls.push_back(&mmul1);
  mmuls.push_back(&mmul2);
  mmuls.push_back(&mmul3);
  mmuls.push_back(&mmul4);

  do_mmul_and_print(mmuls, A, A_T, B, B_T, C, n);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] A_T;
  delete[] B_T;

  return 0;
}
