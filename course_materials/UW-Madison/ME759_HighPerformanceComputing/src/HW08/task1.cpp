#include <iostream>
#include <new>

#include <chrono>
#include <ratio>

#include <functional>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class TimeRecord
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    void execute(std::function<void(const float* A, const float* B, float* C, const std::size_t n)> f, const float* A, const float* B, float* C, const std::size_t n)
    {
        start = high_resolution_clock::now();
        f(A, B, C, n);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    }
    
    auto get_duration() { return duration_sec.count(); }
};

int main(int argc, char* argv[])
{

    // std::function<void()> f_test;

    std::size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    
    omp_set_num_threads(t);

    float *A = new (std::nothrow) float[n * n];
    float *B = new (std::nothrow) float[n * n];
    float *C = new (std::nothrow) float[n * n];

    // assign values of A and B
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = 1;
        B[i] = 1;
    }

    // mmul(A, B, C, n);
    TimeRecord time_record;
    time_record.execute(mmul, A, B, C, n);

    std::cout << C[1] << "\n";
    std::cout << C[n * n - 1] << "\n";
    std::cout << time_record.get_duration() << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
