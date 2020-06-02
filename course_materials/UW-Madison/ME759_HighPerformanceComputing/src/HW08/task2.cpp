#include <iostream>
#include <new>

#include <chrono>
#include <ratio>

#include <functional>

#include "convolution.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class TimeRecord
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    void execute(std::function<void(const float*, float*, std::size_t, const float*, std::size_t)> f, const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
    {
        start = high_resolution_clock::now();
        f(image, output, n, mask, m);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    }
    
    auto get_duration() { return duration_sec.count(); }
};

void assign_input(float *input, std::size_t n) {
    for (std::size_t i = 0; i < n * n; ++i) {
        input[i] = 1;
    }
}

void assign_mask(float *mask, std::size_t m) {
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = 1;
    }
}

int main(int argc, char* argv[])
{
    std::size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);
    
    omp_set_num_threads(t);

    float *input = new (std::nothrow) float[n * n];
    float *output = new (std::nothrow) float[n * n];
    float *mask = new (std::nothrow) float[3 * 3];

    assign_input(input, n);
    assign_mask(mask, 3);

    TimeRecord time_record;
    time_record.execute(Convolve, input, output, n, mask, 3);

    // sample print
    /*
    for (std::size_t i = 0; i < n * n; ++i) {
        std::cout << output[i] << " ";
        if ((i + 1) % n == 0)
            std::cout << "\n";
    }
    */

    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";
    std::cout << time_record.get_duration() << "\n";

    delete[] input;
    delete[] output;
    delete[] mask;

    return 0;
}
