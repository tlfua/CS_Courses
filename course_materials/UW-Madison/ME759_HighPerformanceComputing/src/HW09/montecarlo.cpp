#include "montecarlo.h"

#include <cmath>

int montecarlo(const size_t n, const float *x, const float *y, const float r)
{
    int inside_count = 0;

#pragma omp parallel for reduction(+: inside_count) 
//#pragma omp parallel for simd reduction(+: inside_count) 
    for (size_t i = 0; i < n; ++i){
	inside_count += ((x[i] * x[i] + y[i] * y[i]) < (r * r)) ? 1 : 0;
    }

    return inside_count;
}
