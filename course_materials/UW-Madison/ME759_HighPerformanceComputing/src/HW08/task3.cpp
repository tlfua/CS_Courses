#include <iostream>
#include <new>

#include <chrono>
#include <ratio>

#include <functional>

#include "msort.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class TimeRecord
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    void execute(std::function<void(int*, const std::size_t, const std::size_t)> f, int* arr, const std::size_t n, const std::size_t threshold)
    {
        start = high_resolution_clock::now();
        f(arr, n, threshold);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    }
    
    auto get_duration() { return duration_sec.count(); }
};

void fill_values(int *arr, std::size_t n)
{
    for (int i = 0; i < static_cast<int>(n); ++i) {
        arr[i] = static_cast<int>(n - i - 1);
    }
}

int main(int argc, char* argv[])
{
    std::size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    std::size_t ts = atoi(argv[3]);

    omp_set_num_threads(t);

    int *arr = new int[n];
    fill_values(arr, n);

    TimeRecord time_record;
#pragma omp parallel
    {
#pragma omp single
        {
            time_record.execute(msort, arr, n, ts);

            std::cout << arr[0] << "\n";
            std::cout << arr[n - 1] << "\n";
            std::cout << time_record.get_duration() << "\n";
        }
    }
    
    /*
    // sample print
    for (int i = 0; i < static_cast<int>(n); ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
    */

    delete[] arr;

    return 0;
}
