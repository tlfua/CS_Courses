
from collections import defaultdict
from heapq import heappush, heappop

class Relation:
    def __init__(self):
        self.attributes = []
        self.attribute2values = {}
        self.value_num = 0
        self.keys = []

class Cell:
    def __init__(self, l, weight):
        self.next_level_cells = l
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __gt__(self):
        return self.weight > other.weight

    def __eq__(self):
        return self.weight == other.weight

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, elem):
        heappush(self.heap, elem)

    def pop(self):
        return heappop(self.heap)

    def top(self):
        return self.heap[0]

    def size(self):
        return len(self.heap)

bag = {}

bag[0] = Relation()
bag[0].attributes += ['x', 'y', 'z', 'weight_0']
bag[0].attribute2values['x'] = [1, 1, 1, 1]
bag[0].attribute2values['y'] = [1, 2, 2, 3]
bag[0].attribute2values['z'] = [1, 1, 2, 3]
bag[0].attribute2values['weight_0'] = [1, 1, 1, 1]
bag[0].value_num = 4

bag[1] = Relation()
bag[1].attributes += ['w', 'x', 'y', 'weight_1']
bag[1].attribute2values['w'] = [1, 1]
bag[1].attribute2values['x'] = [1, 1]
bag[1].attribute2values['y'] = [1, 2]
bag[1].attribute2values['weight_1'] = [1, 2]
bag[1].value_num = 2
bag[1].keys += ['x', 'y']

traverse_order = [1, 0]
children = {0: [1], 1: []}
Q = {}

def create_q_input(bag_num, value_iter):
    q_input = []
    for key in bag[bag_num].keys:
        q_input += [key]
        q_input += ['=']
        q_input += [bag[bag_num].attribute2values[key][value_iter]]
        q_input += [',']
    q_input.pop()
    return "".join(str(x) for x in q_input)

def create_child_q_input(bag_num, value_iter, child_num):
    child_q_input = []
    for key in bag[child_num].keys:
        child_q_input += [key]
        child_q_input += ['=']
        child_q_input += [bag[bag_num].attribute2values[key][value_iter]]
        child_q_input += [',']
    child_q_input.pop()
    return "".join(str(x) for x in child_q_input)

def preprocess():

    for bag_num in traverse_order:
        Q[bag_num] = defaultdict(lambda : PriorityQueue())

    for bag_num in traverse_order:
        for value_iter in range(bag[bag_num].value_num):
            
            q_input = None
            if len(bag[bag_num].keys) == 0:
                q_input = "root"
            else:
                q_input = create_q_input(bag_num, value_iter)
            print (q_input)

            l = []
            weight = bag[bag_num].attribute2values["weight_" + str(bag_num)][value_iter]
            go_next_value_iter = False
            for child_num in children[bag_num]:
                child_q_input = create_child_q_input(bag_num, value_iter, child_num)
                if child_q_input not in Q[child_num].keys():
                # for a successful join, keys in all children must have the same valuation
                    print ("not found")
                    go_next_value_iter = True
                    break
                print ("found")
                top = Q[child_num][child_q_input].top()
                l.append(top)
                weight += top.weight
            if go_next_value_iter == True:
                continue
            
            print ("weight = " + str(weight))
            Q[bag_num][q_input].push(Cell(l, weight))

if __name__ == "__main__":
    preprocess()
    print (Q[0]["root"].top().weight)
    Q[0]["root"].pop()
    print (Q[0]["root"].top().weight)
    Q[0]["root"].pop()
    print (Q[0]["root"].top().weight)
    print ("the size = " + str(Q[0]["root"].size()))