
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
        points.append(Point(float(parts[0]), float(parts[1])))
    return point_count, points

def parse_output_format(input_data):
    tour_len = None
    tour = []

    lines = input_data.split('\n')
    tour_len = float(lines[0].split()[0])

    for item in lines[1].split():
        tour.append(int(item))
    #print (tour_len)
    #print (tour)
    #os._exit(0)
    return tour_len, tour

def compute_tour_len(tour, points):
    len_ = 0
    for i in range(len(tour)):
        if i!=len(tour)-1:
            len_ += math.sqrt( (points[tour[i]].x - points[tour[i+1]].x)**2 + (points[tour[i]].y - points[tour[i+1]].y)**2 )
    len_ += math.sqrt( (points[tour[len(tour) - 1]].x - points[tour[0]].x)**2 + (points[tour[len(tour) - 1]].y - points[tour[0]].y)**2 )
    return len_

#ParaItem = namedtuple("ParaItem", ['obj_val', 'err_rate', 'tabu_size', 'max_cur_recs'])
class Para():
    def __init__(self):
        # case#5
        #l_5.append(ParaItem(598000, 0.01, 100000, 10000))

        self.ParaList_by_caseNum = {} # case#1 - #6
        #1
        self.ParaList_by_caseNum[1] = []
        self.ParaList_by_caseNum[1].append({'obj_val': 5000, 'err_rate': 0.01, 'tabu_capacity': 1150, 'max_recs':1000})
        #5
        self.ParaList_by_caseNum[5] = []
        self.ParaList_by_caseNum[5].append({'obj_val': 598000, 'err_rate': 0.01, 'tabu_capacity': 100000, 'max_recs':10000})
        self.ParaList_by_caseNum[5].append({'obj_val': 430000, 'err_rate': 0.007, 'tabu_capacity': 1000000, 'max_recs':100000})
        self.ParaList_by_caseNum[5].append({'obj_val': 405000, 'err_rate': 0.007, 'tabu_capacity': 1000000, 'max_recs':100000})
        self.ParaList_by_caseNum[5].append({'obj_val': 370000, 'err_rate': 0.005, 'tabu_capacity': 1000000, 'max_recs':100000})
        #6
        # target: 78478868
        self.ParaList_by_caseNum[6] = []
        self.ParaList_by_caseNum[6].append({'obj_val': 30000000000, 'err_rate': 0.01, 'tabu_capacity': 1000000, 'max_recs':100000})

    def get_para_val(self, _case_num, _obj_val, _para_key):
        ParaList = self.ParaList_by_caseNum[_case_num]

        if _obj_val > ParaList[0]['obj_val']:
            print ("Error, obj_val must be smaller than ParaList's first obj_val")
            os._exit(0)
        elif _obj_val < ParaList[len(ParaList)-1]['obj_val']:
            return ParaList[len(ParaList)-1][_para_key]
        else:
            for i, para in enumerate(ParaList):
                if _obj_val > para['obj_val']:
                    return ParaList[i-1][_para_key]