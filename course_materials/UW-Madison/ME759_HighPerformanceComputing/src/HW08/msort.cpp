#include "msort.h"

#include <algorithm>
#include <iterator>
#include <vector>

void insertion_sort(int* arr, const std::size_t n)
{
    int key, j; 
    for (int i = 1; i < static_cast<int>(n); ++i) {
        key = arr[i]; 
        j = i - 1; 
  
        while (j >= 0 && arr[j] > key) { 
            arr[j + 1] = arr[j]; 
            j = j - 1; 
        } 
        arr[j + 1] = key; 
    }
}

void fill_back_to_arr(const std::vector<int>& v, int* arr)
{
    // # pragma omp parallel for
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        arr[i] = v[i];
    }
}

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    if (n == 1) {
        return;
    }

    
    if (n < threshold) {
        insertion_sort(arr, n);
        return;
    }
    

    auto left_size = n / 2;
    auto right_size = n - left_size;

# pragma omp task firstprivate (arr, left_size)
        msort(arr, left_size, threshold);

# pragma omp task firstprivate (arr, right_size)
        msort(arr + left_size, right_size, threshold);

# pragma omp taskwait


    // test
    // int arr_test[5] = {4, 5, 1, 2, 3};
    // std::inplace_merge(std::begin(arr_test), std::begin(arr_test) + 2, std::end(arr_test));

    std::vector<int> v(arr, arr + n);
    std::inplace_merge(std::begin(v), std::begin(v) + left_size, std::end(v));
    fill_back_to_arr(v, arr);
}
