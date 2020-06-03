#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import math
from random import shuffle
from collections import namedtuple
import numpy as np
from helper_function import parse_input_format, parse_output_format, compute_tour_len
#from helper_function import gen_tabu_key
from helper_function import Para

Point = namedtuple("Point", ['x', 'y'])

class Tabu():
    def __init__(self, _capacity):
        self.capacity = _capacity
        #self.set_ = set()
        self.set_ = dict()

    def set_capacity(self, _capacity):
        self.capacity = _capacity

    def add(self, key):
        if len(self.set_) == self.capacity:
            self.set_.clear()
        self.set_[key] = None

    def find(self, key):
        return key in self.set_.keys()

    def get_size(self):
        return len(self.set_)
#----------------------------------------------------------------------------
class Local_Search():
    def __init__(self, _point_count, _points):
        self.point_count = _point_count
        self.points = _points

        self._distance = None

        if self.point_count != 33810:  # can not grape so many memories for case#6
            self._distance = np.zeros((self.point_count, self.point_count))
            for i in range(self.point_count):
                for j in range(i+1, self.point_count):
                    self._distance[i][j] = math.sqrt( (self.points[i].x - self.points[j].x)**2 + (self.points[i].y - self.points[j].y)**2 )
                    self._distance[j][i] = self._distance[i][j]
            #print (self.distance_[0][1])

        self.tabu_ = None
        self.case_num = None
        if self.point_count == 51:
            self.case_num = 1
        elif self.point_count == 100:
            self.case_num = 2
        elif self.point_count == 200:
            self.case_num = 3
        elif self.point_count == 574:
            self.case_num = 4
        elif self.point_count == 1889:
            self.case_num = 5
        elif self.point_count == 33810:
            self.case_num = 6

        self.opt_tour_len = 0
        self.opt_tour = []
        self.op_mode = input("Please specify the op mode. 0)Develop, 1)Submit, 2)Load sub-optimal file: ")

        if self.op_mode=='0':
            for i in range(self.point_count):
                self.opt_tour.append(i)
            shuffle(self.opt_tour)
            self.opt_tour_len = compute_tour_len(self.opt_tour, self.points)
        elif self.op_mode=='2':
            my_file_location = input("Please enter sub-optimal file location: ")
            with open(my_file_location, 'r') as input_data_file:
                input_data = input_data_file.read()
            self.opt_tour_len, self.opt_tour = parse_output_format(input_data)

        self.cur_tour_len = self.opt_tour_len
        self.cur_tour = self.opt_tour
        print (self.cur_tour_len)
        print (self.opt_tour_len)
        #os._exit(0)
    #----------------------------
    def run(self):
        max_searches_num, stop_criteria = None, None
        T_org, alpha = None, None
        err_rate = 0.01
        print_window = None
        max_recs = 10000

        if   self.case_num == 1:
            max_searches_num, stop_criteria = -1, 430
            T_org, alpha, print_window, self.tabu_ = 0.1, 0.9, 1, Tabu(1150)
        elif self.case_num == 2:
            max_searches_num ,T_org, alpha, err_rate, print_window, self.tabu_ = 10000000, 200, 0.9, 0.005, 200, Tabu(3400)
        elif self.case_num == 3:
            max_searches_num ,T_org, alpha, print_window, self.tabu_ = 100000, 200, 0.9, 200, Tabu(10000)
        elif self.case_num == 4:
            max_searches_num ,T_org, alpha, print_window, self.tabu_ = 10000000, 200, 0.9, 200, Tabu(80000)
        elif self.case_num == 5:
            max_searches_num, stop_criteria = -1, 378069 #7pt
            T_org, alpha, print_window, self.tabu_ = 0.1, 0.9, 1000, Tabu(100000)
        elif self.case_num == 6:
            max_searches_num = -1
            #stop_criteria = 1000000000
            #stop_criteria = 800000000
            #stop_criteria = 600000000
            #stop_criteria = 400000000
            stop_criteria = 200000000
            #stop_criteria = 100000000
            #stop_criteria = 78478868
            T_org, alpha, print_window, self.tabu_ = 0.1, 0.9, 1000, Tabu(100000)

        para = Para()

        T = T_org
        cur_len_recs = []
        reheat = 0
        #for i in range(max_searches_num):
        it = 0
        while True:
            if max_searches_num != -1:
                if it==max_searches_num:
                    break
            elif max_searches_num == -1:
                if self.opt_tour_len <= stop_criteria:
                    break

            self.tabu_.set_capacity(para.get_para_val(self.case_num, self.cur_tour_len, 'tabu_capacity'))

            # re-heat
            cur_len_recs.append(self.cur_tour_len)
            max_recs = para.get_para_val(self.case_num, self.cur_tour_len, 'max_recs')
            if len(cur_len_recs) == max_recs:
                if abs(cur_len_recs[0]-cur_len_recs[max_recs-1]) < 0.01: # the difference is very small
                    reheat = 1
                else:
                    reheat = 0
                del cur_len_recs[:]
            else:
                reheat = 0

            err_rate = para.get_para_val(self.case_num, self.cur_tour_len, 'err_rate')
            self._gen_next_state(T, reheat=reheat, err_rate=err_rate) # update  self.cur_tour & self.cur_tour_len

            if self.cur_tour_len < self.opt_tour_len:
                self.opt_tour_len = self.cur_tour_len
                self.opt_tour = self.cur_tour
            T *= alpha

            if it%print_window == 0:
                #print ("it {}, curV={}, optV={}, T={}, rh={}, TaSize={}".format(it, self.cur_tour_len, self.opt_tour_len, T, reheat, self.tabu_.get_size()))
                print ("it {}, curV={}, optV={}, rh={}, err_rate={}, TaSize={}".format(it, self.cur_tour_len, self.opt_tour_len, reheat, err_rate, self.tabu_.get_size()))
            it += 1
            #break
    #----------------------------
    def _gen_next_state(self, T, reheat=0, err_rate=0.01, bounce_p=0.1):
        est_tour_len, tour_idx_1, tour_idx_2 = self._two_opt_len()
        if tour_idx_1==-1 and tour_idx_2==-1:
            return

        first_point, second_point = min(self.cur_tour[tour_idx_1], self.cur_tour[tour_idx_2]), max(self.cur_tour[tour_idx_1], self.cur_tour[tour_idx_2])
        first_len, second_len = min(self.cur_tour_len, est_tour_len), max(self.cur_tour_len, est_tour_len)

        #self.tabu_.add(str(int(est_tour_len)))
        key_ = str(first_point)+str(second_point)+str(int(first_len))+str(int(second_len))
        self.tabu_.add(key_)

        p = np.random.rand(1)[0]

        if est_tour_len<=self.cur_tour_len:
            self.cur_tour_len = est_tour_len
            self.cur_tour = self._two_opt_swap(tour_idx_1, tour_idx_2)
            return
        elif reheat == 1:
            if (est_tour_len - self.opt_tour_len)/self.opt_tour_len < err_rate:
                if p < bounce_p:
                    self.cur_tour_len = est_tour_len
                    self.cur_tour = self._two_opt_swap(tour_idx_1, tour_idx_2)
        elif p<math.exp(-(est_tour_len-self.cur_tour_len)/ T):
            self.cur_tour_len = est_tour_len
            self.cur_tour = self._two_opt_swap(tour_idx_1, tour_idx_2)
            return
    #----------------------------
    def gen_ans(self):
        my_file_location = None
        output_text = None

        if self.op_mode=='0' or self.op_mode=='2':
            output_text = '%.2f' % self.opt_tour_len + ' ' + str(0) + '\n'
            output_text += ' '.join(map(str, self.opt_tour))

            my_file_location = 'result_#' + str(self.case_num) + '_' + str(int(self.opt_tour_len))
            with open(my_file_location, 'w') as my_output_file:
                my_output_file.write(output_text)
        elif self.op_mode=='1':
            my_file_location = input("Please enter submit file location: ")
            with open(my_file_location, 'r') as my_input_file:
                output_text = my_input_file.read()

        return output_text
    #----------------------------
    def _get_dist(self, i, j):
        if self.point_count != 33810:
            return self._distance[i][j]
        elif self.point_count == 33810:
            return math.sqrt( (self.points[i].x - self.points[j].x)**2 + (self.points[i].y - self.points[j].y)**2 )
    #----------------------------
    def _two_opt_len(self):
        first_, second_ = None, None

        first_ = np.random.choice(self.point_count, 1)[0] # choose first one
        exclude1_, exclude2_ =  (first_-1)%self.point_count, (first_+1)%self.point_count

        second_condidates = []
        for i in range(self.point_count):
            if i!= first_ and i!=exclude1_ and i!=exclude2_:
                second_condidates.append(i)
        second_ = np.random.choice(second_condidates, 1)[0] # choose second one

        first_, second_ = min(first_, second_), max(first_, second_)
        ## ToDo: swap edge
        '''
        org:                   cur_tour[first_] ---- cur_tour[first+1]
                                 cur_tour[second_] ---- cur_tour[second_+1]
        after swap:      cur_tour[first_] ---- cur_tour[second_]
                                 cur_tour[first+1] ---- cur_tour[second_+1]
        '''
        est_tour_len = None
        rm_dist_1 = self._get_dist(self.cur_tour[first_], self.cur_tour[first_+1])
        add_dist_1 = self._get_dist(self.cur_tour[first_], self.cur_tour[second_])
        rm_dist_2 = None
        add_dist_2 = None
        if second_ != self.point_count-1:
            rm_dist_2 = self._get_dist(self.cur_tour[second_], self.cur_tour[second_+1])
            add_dist_2 = self._get_dist(self.cur_tour[first_+1], self.cur_tour[second_+1])
        elif second_ == self.point_count-1:
            rm_dist_2 = self._get_dist(self.cur_tour[second_], self.cur_tour[0])
            add_dist_2 = self._get_dist(self.cur_tour[first_+1], self.cur_tour[0])

        est_tour_len = self.cur_tour_len - rm_dist_1 - rm_dist_2
        est_tour_len +=  (add_dist_1 + add_dist_2)

        #if self.tabu_.find(str(int(est_tour_len))):

        first_point, second_point = min(self.cur_tour[first_], self.cur_tour[second_]), max(self.cur_tour[first_], self.cur_tour[second_])
        first_len, second_len = min(self.cur_tour_len, est_tour_len), max(self.cur_tour_len, est_tour_len)
        key_ = str(first_point)+str(second_point)+str(int(first_len))+str(int(second_len))

        if self.tabu_.find(key_):
            return None, -1, -1
        return est_tour_len, first_, second_
    #----------------------------
    def _two_opt_swap(self, idx_1, idx_2):
        next_tour = []
        for i in range(idx_1+1):
            next_tour.append( self.cur_tour[i] )
        for i in reversed(range(idx_1+1, idx_2+1)):
            next_tour.append( self.cur_tour[i] )
        for i in range(idx_2+1, self.point_count):
            next_tour.append( self.cur_tour[i] )

        return next_tour


#----------------------------------------------------------------------------
def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    point_count, points = parse_input_format(input_data)

    local_search = Local_Search(point_count, points)

    if local_search.op_mode=='0' or local_search.op_mode=='2':
        local_search.run()

    return local_search.gen_ans()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')