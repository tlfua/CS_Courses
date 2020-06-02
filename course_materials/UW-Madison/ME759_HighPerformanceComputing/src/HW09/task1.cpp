#include "cluster.h"

#include <iostream>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <chrono>
#include <functional>
#include <algorithm>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class TimeRecord
{
private:
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    //cluster(n, t, arr, centers, dists);
    void execute(std::function<void(size_t, size_t, const int*, const int*, int*)> f, size_t n, size_t t, const int* arr, const int* centers, int* dists)
    {
        start = high_resolution_clock::now();
        f(n, t, arr, centers, dists);
        end = high_resolution_clock::now();

        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    }
    
    auto get_duration() const { return duration_sec.count(); }
};

void sample_fill_arr(int* arr)
{
    arr[0] = 0; 
    arr[1] = 1;
    arr[2] = 3;
    arr[3] = 4;
    arr[4] = 6;
    arr[5] = 6;
    arr[6] = 7;
    arr[7] = 8; 
}

void general_fill_arr(int* arr, size_t n)
{
    for (size_t i = 0; i < n; ++i){
        arr[i] = 0;
    }
    std::sort(arr, arr + n);
}

void fill_centers(int* centers, size_t t, int m)
{
    for (size_t i = 1; i <= t; ++i){
        centers[i - 1] = m * (2 * i - 1);
	//std::cout << centers[i - 1] << " ";
    }
    //std::cout << "\n";
}

std::pair<int, size_t> get_max_dist_and_id(const int* dists, size_t t)
{
    int max_val = dists[0];
    size_t max_id = 0;
    for (size_t i = 0; i < t; ++i){
        if (dists[i] > max_val){
	    max_val = dists[i];
	    max_id = i;
	}
    }
    return std::pair<int, size_t>(max_val, max_id);
}

void test_run(size_t n, size_t t)
{
    int* arr = new (std::nothrow) int[n];
    int* centers = new (std::nothrow) int[t];
    int* dists = new (std::nothrow) int[t];

    sample_fill_arr(arr);
    fill_centers(centers, t, n / (2*t));
    cluster(n, t, arr, centers, dists);

    // sample print
    for (size_t i = 0; i < t; ++i){
        std::cout << dists[i] << " ";
    }
    std::cout << "\n";

    delete[] arr;
    delete[] centers;
    delete[] dists;
}

int get_ans(int n, int t)
{
    int per_part_num = n / t;
    int m = n / (2 * t);

    int cen = m + (t - 1) * (2 * m);
    return cen * per_part_num;
}

void run(size_t n, size_t t)
{

    int execution_count = 10;
    int count = execution_count;
    double sum = 0;
    while (count > 0) {
        int* arr = new (std::nothrow) int[n];
        int* centers = new (std::nothrow) int[t];
        int* dists = new (std::nothrow) int[t];

	general_fill_arr(arr, n);
        fill_centers(centers, t, n / (2*t));

        TimeRecord time_record;
        time_record.execute(cluster, n, t, arr, centers, dists);
        sum += time_record.get_duration();

        --count;
        if (count == 0) {
	    std::cout << get_max_dist_and_id(dists, t).first << "\n";
	    //std::cout << "ans = " << get_ans(n, t) << "\n";
            std::cout << get_max_dist_and_id(dists, t).second << "\n";
	}

        delete[] arr;
        delete[] centers;
        delete[] dists;
    }
    std::cout << (sum / (double)execution_count) << "\n";
}

int main(int argc, char* argv[])
{
    size_t n = static_cast<size_t>(atoi(argv[1]));
    size_t t = static_cast<size_t>(atoi(argv[2]));

    if ((n == 8) && (t == 2)) {
        test_run(n, t);
    } else {
        run(n, t);
    }

    return 0;    
}
