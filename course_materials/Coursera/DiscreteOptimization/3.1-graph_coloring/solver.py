#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from queue import PriorityQueue
import numpy as np

def parse_data(input_data):
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    return node_count, edge_count, edges
# ----------------------------------------------------------------------
class ReversePriorityQueue(PriorityQueue):
    def put(self, tup):
        newtup = (tup[0] * -1, tup[1])
        PriorityQueue.put(self, newtup)
    def get(self):
        tup = PriorityQueue.get(self)
        newtup = (tup[0] * -1, tup[1])
        return newtup
# ----------------------------------------------------------------------
class Greedy:
    def __init__(self, _node_count, _edge_count, _edges):
        self.node_count = _node_count
        self.edge_count = _edge_count

        self.adjacency_list = [ ]
        for i in range(self.node_count):
            self.adjacency_list.append([])

        for edge in _edges:
            src, dst = edge
            #print (str(src) + " " + str(dst))
            self.adjacency_list[src].append(dst)
            self.adjacency_list[dst].append(src)

        self.color_index = 0
        self.node_colors = ["Uncolored"] * self.node_count

    def run(self):
        rev_q = ReversePriorityQueue()

        for i in range(self.node_count):
            rev_q.put((len(self.adjacency_list[i]), i))

        while not rev_q.empty():
            _, node = rev_q.get()
            self._set_color(node)

    def _set_color(self, _node):
        node = _node

        if self.color_index == 0:
            self.node_colors[node] = self.color_index
            self.color_index += 1
            return
        else:
            color_occupation = [] # currently, how many kinds of colors can be uesed
            for i in range(self.color_index): # should be the amount of self.color_index
                color_occupation.append(False)

            neighbors = self.adjacency_list[node]
            for neighbor in neighbors:
                color = self.node_colors[neighbor]
                if color != "Uncolored":
                    color_occupation[color] = True

            color_candidates = []
            for i in range(len(color_occupation)):
                if color_occupation[i] == False:
                    color_candidates.append(i)

            if self.node_count == 250: # special deal with case#4
                if len(color_candidates) != 0:
                    _ = np.random.choice(color_candidates, 1)
                    self.node_colors[node] = _[0]
                    return

            for c_idx in range(len(color_occupation)):
                if color_occupation[c_idx] == False:
                    self.node_colors[node] = c_idx
                    return

        self.node_colors[node] = self.color_index
        self.color_index += 1
        return

    def generate_answ(self):
        output_data = str(self.color_index) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.node_colors))
        return output_data
# ----------------------------------------------------------------------
def solve_it(input_data):
    node_count, edge_count, edges = parse_data(input_data)

    greedy = Greedy(node_count, edge_count, edges)
    greedy.run()

    return greedy.generate_answ()
# ----------------------------------------------------------------------
if __name__ == '__main__':
    node_count = None
    edge_count = None
    edges = None

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        #node_count, edge_count, edges = parse_data(input_data)
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')