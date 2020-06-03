import sys
import os
import math
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])

def parse_input_format(input_data):
    # parse the input
    lines = input_data.split('\n')
    point_count = int(lines[0])

    points = []
    for i in range(1, point_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[1]), float(parts[2])))
    return point_count, points
# ---------------------------------------------------------------
def compute_squared_dist(point_1, point_2):
    return (point_1.x - point_2.x)**2 + (point_1.y - point_2.y)**2
# ---------------------------------------------------------------
class Nearest_Neighbor():
    def __init__(self, _point_count, _points):
        self.point_count = _point_count
        self.points = _points
        self.tour = [0]  # start from 0 as my own convention
        self.tour_len = 0

        self.unvisited_pt_idx = set()
        for idx in range(1, self.point_count):
            self.unvisited_pt_idx.add(idx)

    def run(self):
        it = 0
        while len(self.unvisited_pt_idx) != 0:
            it += 1
            if it % 10 == 0:
                print ("it={}".format(it))

            last_pt_idx = self.tour[-1]

            min_ = sys.maxsize
            target_pt_idx = None
            for pt_idx in self.unvisited_pt_idx:
                dist = compute_squared_dist(self.points[last_pt_idx], self.points[pt_idx])
                if dist < min_:
                    min_ = dist
                    target_pt_idx = pt_idx

            self.tour_len += math.sqrt(min_)
            self.tour.append(target_pt_idx)
            self.unvisited_pt_idx.remove(target_pt_idx)

        self.tour_len += math.sqrt(compute_squared_dist(self.points[self.tour[self.point_count-1]], self.points[self.tour[0]]))
        print ("cost={}".format(int(self.tour_len)))
# ---------------------------------------------------------------
if __name__ == "__main__":

    input_file  = sys.argv[1]
    input_data = None
    with open(input_file, 'r') as input_file_obj:
        input_data = input_file_obj.read()

    point_count, points = parse_input_format(input_data)

    nn = Nearest_Neighbor(point_count, points)
    nn.run()