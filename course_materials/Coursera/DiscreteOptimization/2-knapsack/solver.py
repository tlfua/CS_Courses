#!/usr/bin/python
# -*- coding: utf-8 -*-

from operator import attrgetter
from collections import namedtuple
import itertools
import sys
import os

Item = namedtuple("Item", ['index', 'value', 'weight', 'ratio'])

def parse_data(input_data):

    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append( Item( i-1, int(parts[0]), int(parts[1]), int(parts[0])/ int(parts[1])))
    return items, item_count, capacity

class Solver:
    def __init__(self, items, item_count, capacity):
        self.items = items
        self.item_count = item_count
        self.capacity = capacity
        self.sorted_items = sorted(self.items, key=attrgetter('ratio'), reverse=True)
        #for item in self.sorted_items:
        # print (item.ratio)
        self.global_opt_val = -1
        self.global_sorted_map = [0] * self.item_count
        self.global_map = [0] * self.item_count

        self.relax_opt_val = 0
        curr_capacity = self.capacity
        for item in self.sorted_items:
            if item.weight <= curr_capacity:
                curr_capacity -= item.weight
                self.relax_opt_val += item.value
            elif item.weight > curr_capacity:
                self.relax_opt_val += item.value * curr_capacity / item.weight
                break

    def _estimate_remained_opt(self, start_idx, _capacity):
        items = self.sorted_items
        capacity = _capacity
        val = 0
        for idx in range(start_idx, self.item_count):
            if items[idx].weight <= capacity:
                capacity -= items[idx].weight
                val += items[idx].value
            elif items[idx].weight > capacity:
                val += items[idx].value * capacity / items[idx].weight
                break
        return val

    def run_DFS(self, _map, _val, _capacity):
        items = self.sorted_items
        curr_idx = len(_map)-1
        step_decision = _map[ curr_idx ]
        val = _val
        capacity = _capacity

        if curr_idx == self.item_count: # base case
            if val > self.global_opt_val:
                self.global_opt_val = val
                self.global_sorted_map = _map[0:curr_idx]
                #print ("cap = " + str(capacity) )
            return

        if step_decision == "1":
            if items[curr_idx].weight <= capacity:
                capacity -= items[curr_idx].weight
                val += items[curr_idx].value
            elif items[curr_idx].weight > capacity:
                return
        elif step_decision == "0":  # estimate
            est_opt_val = val + self._estimate_remained_opt(start_idx=curr_idx+1, _capacity=capacity)
            if est_opt_val < self.global_opt_val:
                return

        self.run_DFS(_map+"1", val, capacity)
        self.run_DFS(_map+"0", val, capacity)

    def generate_answ(self):
        # translate to org map
        for map_idx, map_val in enumerate(self.global_sorted_map):
            if map_val == "1":
                org_idx = self.sorted_items[map_idx].index
                self.global_map[org_idx] = 1

        output_data = str(self.global_opt_val) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.global_map))
        return output_data

def solve_it(input_data):
    sys.setrecursionlimit(1000000)

    items, item_count, capacity = parse_data(input_data)
    solver = Solver(items, item_count, capacity)

    solver.run_DFS(_map="1", _val=0, _capacity=capacity)
    solver.run_DFS(_map="0", _val=0, _capacity=capacity)

    return solver.generate_answ()

if __name__ == '__main__':
    sys.setrecursionlimit(1000000)

    items = None
    item_count = None
    capacity = None

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        #print(solve_it(input_data))
        items, item_count, capacity = parse_data(input_data)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

    solver = Solver(items, item_count, capacity)
    constrain = capacity
    print ("constrain = " + str(constrain))

    solver.run_DFS(_map="1", _val=0, _capacity=capacity)
    solver.run_DFS(_map="0", _val=0, _capacity=capacity)

    print (solver.generate_answ())