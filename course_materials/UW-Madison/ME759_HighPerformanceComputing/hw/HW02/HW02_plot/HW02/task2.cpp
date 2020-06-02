#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ratio>

#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void assign_input(float* input)
{
    float sample_input[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    for (int i = 0; i < 16; i++) {
        input[i] = sample_input[i];
    }
}

void assign_mask(float* mask)
{
    float sample_mask[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
    for (int i = 0; i < 9; i++) {
        mask[i] = sample_mask[i];
    }
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
    float* input = (float*)malloc(n * n * sizeof(float));
    float* output = (float*)malloc(n * n * sizeof(float));
    float* mask = (float*)malloc(3 * 3 * sizeof(float));

    assign_input(input);
    assign_mask(mask);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    Convolve(input, output, n, mask, 3);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    
    std::cout << duration_sec.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n*n - 1] << "\n";

    free(input);
    free(output);
    free(mask);

    return 0;
}