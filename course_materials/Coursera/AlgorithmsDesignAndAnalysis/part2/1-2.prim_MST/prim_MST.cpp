
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>

class Graph
{
  public:
    Graph(int count_node)
    {
        adj_list_.resize(count_node);
    }

    void AddEdge(int src, int tge, int weight)
    {
        adj_list_[src].push_back(std::pair<int, int>(tge, weight));
        adj_list_[tge].push_back(std::pair<int, int>(src, weight));
    }

    const std::vector<std::pair<int, int>>& GetNeighbors(int src)
    {
        return adj_list_[src];
    }

  private:
    std::vector<std::vector<std::pair<int, int>>> adj_list_;
};


struct Edge
{
    int src, tge, weight;
    Edge(int src, int tge, int weight)
      : src(src),
        tge(tge),
        weight(weight)
    {}
};

struct EdgeComparator
{
    bool operator()(const Edge& lhs, const Edge& rhs)
    {
        return lhs.weight > rhs.weight;
    }
};


int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Please specify the input file." << std::endl;
        return -1;
    }

    // Construct the graph and priority queue.
    std::ifstream stream(argv[1]);
    int count_node, count_edge;
    stream >> count_node >> count_edge;

    Graph graph(count_node);
    for (int i = 0 ; i < count_edge ; ++i) {
        int src, tge, weight;
        stream >> src >> tge >> weight;
        graph.AddEdge(src - 1, tge - 1, weight);
    }

    std::vector<bool> vec_cut(count_node);
    for (int i = 1 ; i < count_node ; ++i)
        vec_cut[i] = false;

    // Preprocess before Prim MST.
    std::priority_queue<Edge, std::vector<Edge>, EdgeComparator> queue;

    vec_cut[0] = true;
    auto& neighbors = graph.GetNeighbors(0);
    for (auto neighbor : neighbors) {
        int node = neighbor.first;
        int weight = neighbor.second;
        queue.push(Edge(0, node, weight));
    }

    // Run the Prim MST algorithm.
    int mst = 0;
    for (int round = 1 ; round < count_node ; ++round) {
        int select;
        while (true) {
            auto& edge = queue.top();
            int src = edge.src, tge = edge.tge;
            if (vec_cut[src] == false && vec_cut[tge] == true) {
                select = src;
                mst += edge.weight;
                break;
            }
            if (vec_cut[src] == true && vec_cut[tge] == false) {
                select = tge;
                mst += edge.weight;
                break;
            }
            queue.pop();
        }

        vec_cut[select] = true;
        auto& neighbors = graph.GetNeighbors(select);
        for (auto neighbor : neighbors) {
            int node = neighbor.first;
            if (vec_cut[node] == true)
                continue;
            int weight = neighbor.second;
            queue.push(Edge(select, node, weight));
        }
    }

    std::cout << mst << std::endl;

    return 0;
}