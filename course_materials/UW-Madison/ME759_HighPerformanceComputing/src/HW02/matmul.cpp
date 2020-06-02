#include "matmul.h"

void mmul1(const double *A, const double *B, double *C, const std::size_t n) {
  int len = (int)n;
  for (int A_row_iter = 0; A_row_iter < len; A_row_iter++) {
    for (int B_col_iter = 0; B_col_iter < len; B_col_iter++) {
      C[A_row_iter * n + B_col_iter] = 0;
      for (int k = 0; k < len; k++) {
        C[A_row_iter * n + B_col_iter] +=
            A[A_row_iter * n + k] * B[B_col_iter + k * n];
      }
    }
  }
}

void mmul2(const double *A, const double *B, double *C, const std::size_t n) {
  int len = (int)n;
  for (int B_col_iter = 0; B_col_iter < len; B_col_iter++) {
    for (int A_row_iter = 0; A_row_iter < len; A_row_iter++) {
      C[A_row_iter * n + B_col_iter] = 0;
      for (int k = 0; k < len; k++) {
        C[A_row_iter * n + B_col_iter] +=
            A[A_row_iter * n + k] * B[B_col_iter + k * n];
      }
    }
  }
}

void mmul3(const double *A, const double *B_T, double *C, const std::size_t n) {
  int len = (int)n;
  for (int A_row_iter = 0; A_row_iter < len; A_row_iter++) {
    for (int B_col_iter = 0; B_col_iter < len; B_col_iter++) {
      C[A_row_iter * n + B_col_iter] = 0;
      for (int k = 0; k < len; k++) {
        C[A_row_iter * n + B_col_iter] +=
            A[A_row_iter * n + k] * B_T[B_col_iter * n + k];
      }
    }
  }
}

void mmul4(const double *A_T, const double *B, double *C, const std::size_t n) {
  int len = (int)n;
  for (int A_row_iter = 0; A_row_iter < len; A_row_iter++) {
    for (int B_col_iter = 0; B_col_iter < len; B_col_iter++) {
      C[A_row_iter * n + B_col_iter] = 0;
      for (int k = 0; k < len; k++) {
        C[A_row_iter * n + B_col_iter] +=
            A_T[A_row_iter + k * n] * B[B_col_iter + k * n];
      }
    }
  }
}
