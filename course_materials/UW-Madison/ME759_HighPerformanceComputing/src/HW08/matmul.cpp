#include "matmul.h"

#include <iostream>

void mmul(const float *A, const float *B_T, float *C, const std::size_t n)
{
	// std::cout << omp_get_num_threads() << "\n";

#pragma omp parallel
{		
    // std::cout << omp_get_num_threads() << "\n";

    # pragma omp for
    for (std::size_t A_row_iter = 0; A_row_iter < n; ++A_row_iter) {
        for (std::size_t B_col_iter = 0; B_col_iter < n; ++B_col_iter) {
            C[A_row_iter * n + B_col_iter] = 0;
            for (std::size_t k = 0; k < n; ++k) {
                C[A_row_iter * n + B_col_iter] +=
                    A[A_row_iter * n + k] * B_T[B_col_iter * n + k];
            }
        }
    }
}
}
