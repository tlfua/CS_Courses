#include "matrix.h"
#include "vector.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
// #include <ratio>
#include <functional>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

using namespace sequential;

class TimeRecord
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    void execute(std::function<void(matrix&, vector&, vector&, vector&, int&, double)> f, matrix& A, vector& b, vector& x0, vector& x, int& iter_count)
    {
        start = high_resolution_clock::now();
        f(A, b, x0, x, iter_count, 1e-6);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    }

    auto get_duration() { return duration_sec.count(); }
};

void fill_matrix(matrix& A)
{
    for (int i = 1; i <= rows(A); ++i){
        for (int j = 1; j <= columns(A); ++j){
            A(i, j) = ((double)rand() / (double) RAND_MAX);
        }
    }
}

void fill_vector(vector& b)
{
    for (int i = 1; i <= rows(b); ++i){
        b(i) = 1;
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
