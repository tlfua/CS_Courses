import queue
from collections import namedtuple

Edge = namedtuple('Edge', ['vertex', 'weight'])
#vertices_count = 200

class GraphUndirectedWeighted(object):
    def __init__(self, vertex_count):
        self.vertex_count = vertex_count
        self.adjacency_list = [[] for _ in range(vertex_count)]

    def add_edge(self, source, dest, weight):
        #assert source < self.vertex_count
        #assert dest < self.vertex_count
        self.adjacency_list[source].append(Edge(dest, weight))
        #self.adjacency_list[dest].append(Edge(source, weight))

    def get_edge(self, vertex):
        for e in self.adjacency_list[vertex]:
            yield e

    def get_vertex(self):
        for v in range(self.vertex_count):
            yield v


def dijkstra(graph, source, dest):
    q = queue.PriorityQueue()
    parents = []
    distances = []
    start_weight = float("inf")

    for i in graph.get_vertex():
        weight = start_weight
        if source == i:
            weight = 0
        distances.append(weight)
        parents.append(None)

    q.put(([0, source]))

    while not q.empty():
        v_tuple = q.get()
        v = v_tuple[1]

        for e in graph.get_edge(v):
            candidate_distance = distances[v] + e.weight
            if distances[e.vertex] > candidate_distance:
                distances[e.vertex] = candidate_distance
                parents[e.vertex] = v
                # primitive but effective negative cycle detection
                if candidate_distance < -1000:
                    raise Exception("Negative cycle detected")
                q.put(([distances[e.vertex], e.vertex]))

    shortest_path = []
    end = dest
    while end is not None:
        shortest_path.append(end)
        end = parents[end]

    shortest_path.reverse()

    return shortest_path, distances[dest]


def main():
    #g = GraphUndirectedWeighted(9)
    #g.add_edge(0, 1, 4)

    g = GraphUndirectedWeighted(200)

    # Parse file data to Graph structure
    input_data = None
    with open('dijkstraData.txt', 'r') as input_data_file:
        input_data = input_data_file.read()

    lines = input_data.split('\n')
    for line in lines:
        items = line.split()
        #print(items)
        src = int(items[0]) - 1

        for i in range(1, len(items)):
            item = items[i].split(',')
            dst = int(item[0]) - 1
            w = int(item[1])
            #print (src)
            #print (dst)
            #print (w)
            g.add_edge(src, dst, w)
        #break

    #shortest_path, distance = dijkstra(g, 0, 1)
    print (dijkstra(g, 1 - 1, 7 - 1)[1])
    print (dijkstra(g, 1 - 1, 37 - 1)[1])
    print (dijkstra(g, 1 - 1, 59 - 1)[1])
    print (dijkstra(g, 1 - 1, 82 - 1)[1])
    print (dijkstra(g, 1 - 1, 99 - 1)[1])
    print (dijkstra(g, 1 - 1, 115 - 1)[1])
    print (dijkstra(g, 1 - 1, 133 - 1)[1])
    print (dijkstra(g, 1 - 1, 165 - 1)[1])
    print (dijkstra(g, 1 - 1, 188 - 1)[1])
    print (dijkstra(g, 1 - 1, 197 - 1)[1])

if __name__ == "__main__":
    main()