#include "scan.h"

void Scan(const float *arr, float *output, std::size_t n)
{
    output[0] = 0.0;
    int len = (int)n;
    for (int i = 1; i < len; i++) {
        output[i] = output[i - 1] + arr[i - 1];
    }
}
