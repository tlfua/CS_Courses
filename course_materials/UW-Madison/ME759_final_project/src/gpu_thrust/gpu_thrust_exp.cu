#include "matrix.cuh"
#include "vector.cuh"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
// #include <ratio>
#include <functional>

using namespace gpu_thrust;

class TimeRecord
{
private:
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;

public:
    TimeRecord()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~TimeRecord()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void execute(std::function<void(matrix&, vector&, vector&, vector&, int&, double)> f, matrix& A, vector& b, vector& x0, vector& x, int& k)
    {
        cudaEventRecord(start);
        f(A, b, x0, x, k, 1e-6);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    }

    auto get_duration() { return ms; }
};

void fill_matrix(matrix& A)
{
    for (int i = 1; i <= rows(A); ++i){
        for (int j = 1; j <= columns(A); ++j){
            A.set(i, j, (double)rand() / (double) RAND_MAX);
        }
    }
}

void fill_vector(vector& b)
{
    for (int i = 1; i <= rows(b); ++i){
        b.set(i, 1);
    }
}

void  run(int n)
{
    matrix A(n, n);
    fill_matrix(A);
    
    vector b(n);
    fill_vector(b);

    vector x0(n);
    vector x(n);
    int iter_count;

    TimeRecord time_record;
    time_record.execute(GMRES, A, b, x0, x, iter_count);
    
    std::cout << "execution time: " << time_record.get_duration() << "\n";
    std::cout << "iteration count: " << iter_count << "\n";
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
    
    srand(time(NULL));
    run(n);

    return 0;
}
