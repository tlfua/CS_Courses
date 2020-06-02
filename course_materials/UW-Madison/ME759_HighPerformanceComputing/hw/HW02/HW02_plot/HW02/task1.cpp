#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ratio>

#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
    float* input = (float*)malloc(n * sizeof(float));
    float* output = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        input[i] = (float)(i + 1)/ (float)(n + 1);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    start = high_resolution_clock::now();
    Scan(input, output, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);

    std::cout << duration_sec.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n - 1] << "\n";

    free(input);
    free(output);

    return 0;
}