
from collections import defaultdict
from heapq import heappush, heappop

class Relation:
    def __init__(self):
        self.attributes = []
        self.attribute2values = {}
        self.value_num = 0
        self.keys = []

class Cell:
    def __init__(self, attribute2value, l, weight):
        self.attribute2value = attribute2value
        self.next_level_cells = l
        self.next_cell = None
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __eq__(self, other):
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
bag[0].attributes += ['x', 'y', 'weight_0']
bag[0].attribute2values['x'] = [1, 2]
bag[0].attribute2values['y'] = [1, 1]
bag[0].attribute2values['weight_0'] = [1, 2]
bag[0].value_num = 2

bag[1] = Relation()
bag[1].attributes += ['y', 'z', 'weight_1']
bag[1].attribute2values['y'] = [1, 3]
bag[1].attribute2values['z'] = [1, 1]
bag[1].attribute2values['weight_1'] = [1, 1]
bag[1].value_num = 2
bag[1].keys += ['y']

bag[2] = Relation()
bag[2].attributes += ['z', 'w', 'weight_2']
bag[2].attribute2values['z'] = [1, 1]
bag[2].attribute2values['w'] = [1, 2]
bag[2].attribute2values['weight_2'] = [1, 4]
bag[2].value_num = 2
bag[2].keys += ['z']

bag[3] = Relation()
bag[3].attributes += ['z', 'u', 'weight_3']
bag[3].attribute2values['z'] = [1, 1]
bag[3].attribute2values['u'] = [1, 2]
bag[3].attribute2values['weight_3'] = [1, 5]
bag[3].value_num = 2
bag[3].keys += ['z']

traverse_order = [2, 3, 1, 0]
children = {0: [1],\
            1: [2, 3],\
            2: [],\
            3: []}
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

def get_attribute2value(bag_num, value_iter):
    attribute2value = {}
    for attribute in bag[bag_num].attribute2values.keys():
        attribute2value[attribute] = bag[bag_num].attribute2values[attribute][value_iter]
    return attribute2value

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
            attribute2value = get_attribute2value(bag_num, value_iter)
            Q[bag_num][q_input].push(Cell(attribute2value, l, weight))

def create_q_input_topdown(cell, bag_num):
    child_q_input = []
    for key in bag[bag_num].keys:
        child_q_input += [key]
        child_q_input += ['=']
        child_q_input += [cell.attribute2value[key]]
        child_q_input += [',']
    child_q_input.pop()
    return "".join(str(x) for x in child_q_input)

def topdown(cell, bag_num):
    # print ("topdown bag_num = " + str(bag_num))

    q_input = None
    if len(bag[bag_num].keys) == 0:
        q_input = "root"
    else:
        q_input = create_q_input_topdown(cell, bag_num)

    if cell.next_cell is None:
        # print ("A")
        Q[bag_num][q_input].pop()
        for i, child_num in enumerate(children[bag_num]):
            old_next_level_cell_weight = cell.next_level_cells[i].weight
            new_next_level_cell = topdown(cell.next_level_cells[i], child_num)
            # print (str(bag_num) + " B")
            if new_next_level_cell is not None:
                # print (str(bag_num) + " D")
                l = cell.next_level_cells[:]
                l[i] = new_next_level_cell
                new_weight = cell.weight - old_next_level_cell_weight + new_next_level_cell.weight
                Q[bag_num][q_input].push(Cell(cell.attribute2value, l, new_weight))
        # print ("C")
        if q_input != "root":
            cell.next_cell = Q[bag_num][q_input].top()
    return cell.next_cell

def enumeration():
    ret = Q[0]["root"].top()
    topdown(Q[0]["root"].top(), 0)
    return ret

if __name__ == "__main__":
    preprocess()
    print ("Q[0][root] top weight = " + str(Q[0]["root"].top().weight))
    # Q[0]["root"].pop()
    # print (Q[0]["root"].top().weight)
    # Q[0]["root"].pop()
    # print (Q[0]["root"].top().weight)
    # print ("the size = " + str(Q[0]["root"].size()))

    ret = enumeration()
    print ("after 1 enumeration")
    print ("Q[0][root] " + str(Q[0]["root"].size()))
    print ("Q[1][y=1] " + str(Q[1]["y=1"].size()))
    print ("Q[2][z=1] " + str(Q[2]["z=1"].size()))
    print ("Q[2][z=1] top weight " + str(Q[2]["z=1"].top().weight))
    print ("Q[3][z=1] " + str(Q[3]["z=1"].size()))
    print ("Q[3][z=1] top weight " + str(Q[3]["z=1"].top().weight))