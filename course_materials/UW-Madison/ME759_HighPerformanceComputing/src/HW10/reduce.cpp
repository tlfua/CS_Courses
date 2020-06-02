// Author: Lijing Yang

#include <cstdlib>
#include <iostream>
#include "reduce.h"
#include <omp.h>

float reduce(const float* arr, const size_t l, const size_t r){

  double sum = 0;
  #pragma omp parallel for simd reduction(+:sum)
  for(size_t i = l; i < r; i++) {
     sum += arr[i]; 
  }
  return sum;
}


