import sys
import os

def compute_target_count(input_data, start_t=None, _size=None):
    lines = input_data.split('\n')

    nums_count = {}
    for line in lines:
        num = int(line)
        if num not in nums_count.keys():
            nums_count[num] = 1
        else:
            nums_count[num] +=  1

    x = None
    t_subtract_x = None
    t_count = 0
    for t in range(start_t, start_t + _size):
        for x in nums_count.keys():
            if nums_count[x] > 1:
                continue

            t_subtract_x = t - x
            if (t_subtract_x in nums_count.keys()) and (nums_count[t_subtract_x] == 1):
                t_count += 1
                break

        if t % 10 == 0:
            print ("t={}, t_count={}, x={}, t-x={}".format(t, t_count, x, t_subtract_x))
    print (t_count)

if __name__ == "__main__":

    input_file  = sys.argv[1]
    input_data = None
    with open(input_file, 'r') as input_file_obj:
        input_data = input_file_obj.read()

    compute_target_count(input_data, start_t=-10000, _size=20001)