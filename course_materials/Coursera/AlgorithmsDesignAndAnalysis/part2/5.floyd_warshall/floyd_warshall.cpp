#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <climits>


class Graph
{
  public:
    Graph(const char* name_file)
      : name_file_(name_file)
    {}

    int Solve()
    {
        Init();
        return FloydWarshall();
    }

    static constexpr const int HAS_NEG_CLE = -1;

  private:
    int size_;
    std::string name_file_;
    std::vector<std::vector<int>> dp_;

    void Init();
    int FloydWarshall();
};

void Graph::Init()
{
    std::fstream stream(name_file_.c_str());
    int count_node, count_edge;
    stream >> count_node >> count_edge;

    size_ = count_node;
    dp_.resize(count_node);
    for (int i = 0 ; i < count_node ; ++i) {
        dp_[i].resize(count_node);
        for (int j = 0 ; j < count_node ; ++j)
            dp_[i][j] = INT_MAX;
    }

    for (int i = 0 ; i < count_edge ; ++i) {
        int src, dst, weight;
        stream >> src >> dst >> weight;
        --src;
        --dst;
        dp_[src][dst] = weight;
    }

    for (int i = 0 ; i < count_node ; ++i)
        dp_[i][i] = 0;
}

int Graph::FloydWarshall()
{
    for (int k = 0 ; k < size_ ; ++k) {
        for (int i = 0 ; i < size_ ; ++i) {
            for (int j = 0 ; j < size_ ; ++j) {
                if (dp_[i][k] == INT_MAX || dp_[k][j] == INT_MAX)
                    continue;
                int relax = dp_[i][k] + dp_[k][j];
                if (relax < dp_[i][j])
                    dp_[i][j] = relax;
            }
        }
    }

    for (int i = 0 ; i < size_ ; ++i) {
        if (dp_[i][i] < 0)
            return HAS_NEG_CLE;
    }

    int shortest = INT_MAX;
    for (int i = 0 ; i < size_ ; ++i) {
        for (int j = 0 ; j < size_ ; ++j) {
            if (dp_[i][j] < shortest)
                shortest = dp_[i][j];
        }
    }

    return shortest;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Please specify the 3 input graphs" << std::endl;
        return -1;
    }

    int min = Graph::HAS_NEG_CLE;
    {
        std::cout << "Graph 1 processing ..." << std::endl;
        Graph dp_fst(argv[1]);
        int path = dp_fst.Solve();
        if (path != Graph::HAS_NEG_CLE && path < min)
            min = path;
        std::cout << "Result " << path << std::endl;
    }
    {
        std::cout << "Graph 2 processing ..." << std::endl;
        Graph dp_snd(argv[2]);
        int path = dp_snd.Solve();
        if (path != Graph::HAS_NEG_CLE && path < min)
            min = path;
        std::cout << "Result " << path << std::endl;
    }
    {
        std::cout << "Graph 3 processing ..." << std::endl;
        Graph dp_trd(argv[3]);
        int path = dp_trd.Solve();
        if (path != Graph::HAS_NEG_CLE && path < min)
            min = path;
        std::cout << "Result " << path << std::endl;
    }
    std::cout << min << std::endl;

    return 0;
}