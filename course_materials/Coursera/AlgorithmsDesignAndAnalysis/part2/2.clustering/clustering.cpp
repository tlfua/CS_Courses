#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

struct Edge
{
    int src, tge, dist;
    Edge(int src, int tge, int dist)
      : src(src),
        tge(tge),
        dist(dist)
    {}
};


class DisjointSet
{
  public:
    DisjointSet(int count_node)
      : count_node_(count_node),
        set_count_(count_node)
    {
        parent_.resize(count_node + 1);
        rank_.resize(count_node + 1);
        for (int i = 1 ; i <= count_node ; ++i) {
            parent_[i] = i;
            rank_[i] = 0;
        }
    }

    int Find(int set_num)
    {
        if (set_num != parent_[set_num])
            parent_[set_num] = Find(parent_[set_num]);
        return parent_[set_num];
    }

    void Union(int set_src, int set_tge)
    {
        --set_count_;
        int rank_src = rank_[set_src];
        int rank_tge = rank_[set_tge];

        if (rank_src == rank_tge) {
            parent_[set_src] = set_tge;
            ++rank_[set_tge];
            return;
        }

        if (rank_src < rank_tge)
            parent_[set_src] = set_tge;
        else
            parent_[set_tge] = set_src;
    }

    int GetSetCount()
    {
        return set_count_;
    }

  private:
    int count_node_;
    int set_count_;
    std::vector<int> parent_;
    std::vector<int> rank_;
};


bool SortByDistance(const Edge& lhs, const Edge& rhs)
{
    return lhs.dist < rhs.dist;
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Please specify the input file and cluster count k." << std::endl;
        return -1;
    }

    // Initialize the disjoint set.
    std::ifstream stream(argv[1]);
    int count_node;
    stream >> count_node;
    DisjointSet disjoint_set(count_node);

    // Sort the edges by distances in ascending order.
    std::vector<Edge> vec_edge;
    while (!stream.eof()) {
        int src, tge, dist;
        stream >> src >> tge >> dist;
        vec_edge.push_back(Edge(src, tge, dist));
    }
    std::sort(vec_edge.begin(), vec_edge.end(), SortByDistance);

    // Run the Kruskal algorithm.
    int count_cluster = atoi(argv[2]);
    int idx_edge = 0;
    while (disjoint_set.GetSetCount() > count_cluster) {
        Edge& edge = vec_edge[idx_edge];
        int set_src = disjoint_set.Find(edge.src);
        int set_tge = disjoint_set.Find(edge.tge);
        if (set_src != set_tge)
            disjoint_set.Union(set_src, set_tge);
        ++idx_edge;
    }

    // Output the maximum spacing.
    int max_space;
    for (auto& edge : vec_edge) {
        int set_src = disjoint_set.Find(edge.src);
        int set_tge = disjoint_set.Find(edge.tge);
        if (set_src != set_tge) {
            max_space = edge.dist;
            break;
        }
    }
    std::cout << max_space << std::endl;

    return 0;
}