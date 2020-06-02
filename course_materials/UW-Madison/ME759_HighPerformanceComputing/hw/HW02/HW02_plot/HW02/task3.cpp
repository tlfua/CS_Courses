#include <iostream>
#include <chrono>
#include <ratio>
#include <vector>

#include "matmul.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void transpose(const double* A, double* A_T, const std::size_t n)
{
    int len = (int)n*n;
    for (int i = 0; i < len; i++) {
        int row_iter = i/ n;
        int col_iter = i % n;
        A_T[col_iter*n + row_iter] = A[i];
    }
}

void do_mmul_and_print(const std::vector<void (*)(const double*, const double*, double*, const std::size_t)>& mmuls, double* A, double* A_T, double* B, double* B_T, double* C, const std::size_t n)
{
    double* first_mat;
    double* second_mat;
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

        duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
        std::cout << duration_sec.count() << "\n";
 
        std::cout << C[n*n - 1] << "\n";
    }
}

int main()
{
    int n = 1000;
    std::cout << n << "\n";
    
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    double* A_T = (double*)malloc(n * n * sizeof(double));
    double* B_T = (double*)malloc(n * n * sizeof(double));

    for (int i = 0; i < n*n; i++) {
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

    std::vector<void (*)(const double*, const double*, double*, const std::size_t)> mmuls;
    mmuls.push_back(&mmul1);
    mmuls.push_back(&mmul2);
    mmuls.push_back(&mmul3);
    mmuls.push_back(&mmul4);

    do_mmul_and_print(mmuls, A, A_T, B, B_T, C, n);

    free(A);
    free(A_T);
    free(B);
    free(B_T);
    free(C);

    return 0;
}
