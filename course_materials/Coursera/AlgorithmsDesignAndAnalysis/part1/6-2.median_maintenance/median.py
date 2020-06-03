import sys
import os
from queue import PriorityQueue

class MyPriorityQueue(PriorityQueue):
    def peek_top_priority(self):
        return self.queue[0]
# ---------------------------------------------------------------
class MyReversePriorityQueue(PriorityQueue):
    def put(self, num):
        num *= -1
        PriorityQueue.put(self, num)
    def get(self):
        num = PriorityQueue.get(self)
        return num * -1
    def peek_top_priority(self):
        return -self.queue[0]
# ---------------------------------------------------------------
def step_get_median(max_heap, min_heap):
    if max_heap.qsize() >= min_heap.qsize():
        return max_heap.peek_top_priority()
    else:
        return min_heap.peek_top_priority()
# ---------------------------------------------------------------
def get_medians_sum(input_data):
    lines = input_data.split('\n')

    max_heap = MyReversePriorityQueue() # L half
    min_heap = MyPriorityQueue() # R half
    target_q, other_q = None, None
    median = None
    medians = []

    ## Allow max_heap and min_heap to seperatedly contain 1 element
    first_num, second_num = int(lines[0]), int(lines[1])

    max_heap.put(min(first_num, second_num))
    min_heap.put(max(first_num, second_num))

    medians.append(first_num)
    medians.append(step_get_median(max_heap, min_heap))

    for i in range(2, len(lines)):
        num = int(lines[i])
        # belong
        if num <= max_heap.peek_top_priority():
            target_q = max_heap
            other_q = min_heap
        elif num >= min_heap.peek_top_priority():
            target_q = min_heap
            other_q = max_heap
        else:
            if max_heap.qsize() <= min_heap.qsize():
                target_q = max_heap
                other_q = min_heap
            else:
                target_q = min_heap
                other_q = max_heap

        target_q.put(num)

        if target_q.qsize() - other_q.qsize() == 2:
            other_q.put(target_q.get())
        median = step_get_median(max_heap, min_heap)
        medians.append(median)

    print (sum(medians) % 10000)
# ---------------------------------------------------------------
if __name__ == "__main__":

    input_file  = sys.argv[1]
    input_data = None
    with open(input_file, 'r') as input_file_obj:
        input_data = input_file_obj.read()

    get_medians_sum(input_data)