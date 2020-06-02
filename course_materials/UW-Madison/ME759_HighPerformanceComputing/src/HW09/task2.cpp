#include "montecarlo.h"

#include <iostream>
//#include <new>
#include <ctime>
#include <chrono>
#include <functional>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class TimeRecord
{
private:
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
public:
    int execute(std::function<int(const size_t, const float*, const float*, const float)> f, const size_t n, const float* x, const float* y, const float r)
    {
        start = high_resolution_clock::now();
        int res = f(n, x, y, r);
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
        return res;
    }
    
    auto get_duration() const { return duration_sec.count(); }
};

void fill_input(float* x, size_t n, const float r)
{
    for (size_t i = 0; i < n; ++i){
        x[i] = (-r) + ((float) rand() / (float) RAND_MAX) * (2 * r);
    }
}

void run(size_t n, size_t t)
{
    int execution_count = 10;
    int count = execution_count;
    double sum = 0;
    while (count > 0) {
        float x[1000000];
        float y[1000000];

        srand(time(NULL));
        fill_input(x, n, 1.0);
        fill_input(y, n, 1.0);

        TimeRecord time_record;
        int count_inside = time_record.execute(montecarlo, n, x, y, 1.0);
	sum += time_record.get_duration();
        float pi = 4 * ((float) count_inside / (float) n);

	--count;
	if (count == 0) {
            std::cout << pi << "\n";
	}
    }
    std::cout << (sum / (double)execution_count) << "\n";
}

int main(int argc, char* argv[])
{
    size_t n = atoi(argv[1]);
    int t = atoi(argv[2]);

    omp_set_num_threads(t);

    run(n ,t);

    return 0;
}
