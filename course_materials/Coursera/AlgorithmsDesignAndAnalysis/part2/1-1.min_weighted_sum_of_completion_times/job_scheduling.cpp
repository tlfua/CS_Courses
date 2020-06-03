#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>


enum {
    HEUR_DIFFERENCE = 0,
    HEUR_RATIO = 1
};


struct Job
{
  public:
    int weight, length;
    Job(int weight, int length)
      : weight(weight),
        length(length)
    {}
};


bool SortByDifference(const Job& lhs, const Job& rhs)
{
    int diff_lhs = lhs.weight - lhs.length;
    int diff_rhs = rhs.weight - rhs.length;
    return (diff_lhs != diff_rhs)? (diff_lhs > diff_rhs) : (lhs.weight > rhs.weight);
}


bool SortByRatio(const Job& lhs, const Job& rhs)
{
    double rat_lhs = static_cast<double>(lhs.weight) / lhs.length;
    double rat_rhs = static_cast<double>(rhs.weight) / rhs.length;
    return rat_lhs > rat_rhs;
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Please specify the heuristics and the input file." << std::endl;
        return -1;
    }

    int heur = atoi(argv[1]);
    char* input = argv[2];

    std::ifstream stream(input);
    int count_job;
    stream >> count_job;

    // Read in the jobs.
    std::vector<Job> vec_job;
    for (int i = 0 ; i < count_job ; ++i) {
        int weight, length;
        stream >> weight >> length;
        vec_job.push_back(Job(weight, length));
    }

    // Sort the jobs by different heuristics.
    if (heur == HEUR_DIFFERENCE)
        std::sort(vec_job.begin(), vec_job.end(), SortByDifference);
    else if (heur == HEUR_RATIO)
        std::sort(vec_job.begin(), vec_job.end(), SortByRatio);

    long accu_time = 0, cost = 0;
    for (auto job : vec_job) {
        accu_time += job.length;
        cost += accu_time * job.weight;
    }
    std::cout << cost << std::endl;

    return 0;
}