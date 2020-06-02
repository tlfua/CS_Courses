
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

class RankEngine:
    def __init__(self):
        self.bag = {}

        self.bag[0] = Relation()
        self.bag[0].attributes += ['x', 'y', 'weight_0']
        self.bag[0].attribute2values['x'] = [1, 2]
        self.bag[0].attribute2values['y'] = [1, 1]
        self.bag[0].attribute2values['weight_0'] = [1, 2]
        self.bag[0].value_num = 2

        self.bag[1] = Relation()
        self.bag[1].attributes += ['y', 'z', 'weight_1']
        self.bag[1].attribute2values['y'] = [1, 3]
        self.bag[1].attribute2values['z'] = [1, 1]
        self.bag[1].attribute2values['weight_1'] = [1, 1]
        self.bag[1].value_num = 2
        self.bag[1].keys += ['y']

        self.bag[2] = Relation()
        self.bag[2].attributes += ['z', 'w', 'weight_2']
        self.bag[2].attribute2values['z'] = [1, 1]
        self.bag[2].attribute2values['w'] = [1, 2]
        self.bag[2].attribute2values['weight_2'] = [1, 4]
        self.bag[2].value_num = 2
        self.bag[2].keys += ['z']

        self.bag[3] = Relation()
        self.bag[3].attributes += ['z', 'u', 'weight_3']
        self.bag[3].attribute2values['z'] = [1, 1]
        self.bag[3].attribute2values['u'] = [1, 2]
        self.bag[3].attribute2values['weight_3'] = [1, 5]
        self.bag[3].value_num = 2
        self.bag[3].keys += ['z']

        self.traverse_order = [2, 3, 1, 0]
        self.children = {0: [1],\
                            1: [2, 3],\
                            2: [],\
                            3: []}
        
        self.Q = {}
        for bag_num in self.traverse_order:
            self.Q[bag_num] = defaultdict(lambda : PriorityQueue())

    def preprocess(self):
        for bag_num in self.traverse_order:
            for value_iter in range(self.bag[bag_num].value_num):
            
                q_input = None
                if len(self.bag[bag_num].keys) == 0:
                    q_input = "root"
                else:
                    q_input = self.create_q_input(bag_num, value_iter)
                # print (q_input)

                l = []
                weight = self.bag[bag_num].attribute2values["weight_" + str(bag_num)][value_iter]
                go_next_value_iter = False
                for child_num in self.children[bag_num]:
                    child_q_input = self.create_child_q_input(bag_num, value_iter, child_num)
                    if child_q_input not in self.Q[child_num].keys():
                    # for a successful join, keys in all children must have the same valuation
                        # print ("not found")
                        go_next_value_iter = True
                        break
                    # print ("found")
                    top = self.Q[child_num][child_q_input].top()
                    l.append(top)
                    weight += top.weight
                if go_next_value_iter == True:
                    continue
            
                # print ("weight = " + str(weight))
                attribute2value = self.get_attribute2value(bag_num, value_iter)
                self.Q[bag_num][q_input].push(Cell(attribute2value, l, weight))

    def create_q_input(self, bag_num, value_iter):
        q_input = []
        for key in self.bag[bag_num].keys:
            q_input += [key]
            q_input += ['=']
            q_input += [self.bag[bag_num].attribute2values[key][value_iter]]
            q_input += [',']
        q_input.pop()
        return "".join(str(x) for x in q_input)

    def create_child_q_input(self, bag_num, value_iter, child_num):
        child_q_input = []
        for key in self.bag[child_num].keys:
            child_q_input += [key]
            child_q_input += ['=']
            child_q_input += [self.bag[bag_num].attribute2values[key][value_iter]]
            child_q_input += [',']
        child_q_input.pop()
        return "".join(str(x) for x in child_q_input)

    def get_attribute2value(self, bag_num, value_iter):
        attribute2value = {}
        for attribute in self.bag[bag_num].attribute2values.keys():
            attribute2value[attribute] = self.bag[bag_num].attribute2values[attribute][value_iter]
        return attribute2value

    def enumeration(self):
        ret = self.Q[0]["root"].top()
        self.topdown(self.Q[0]["root"].top(), 0)
        return ret

    def topdown(self, cell, bag_num):
    # print ("topdown bag_num = " + str(bag_num))

        q_input = None
        if len(self.bag[bag_num].keys) == 0:
            q_input = "root"
        else:
            q_input = self.create_q_input_topdown(cell, bag_num)

        if cell.next_cell is None and self.Q[bag_num][q_input].size() > 0:
        # if cell.next_cell is None:  # not connect yet
            self.Q[bag_num][q_input].pop()
            for i, child_num in enumerate(self.children[bag_num]):
                old_next_level_cell_weight = cell.next_level_cells[i].weight
                new_next_level_cell = self.topdown(cell.next_level_cells[i], child_num)
                if new_next_level_cell is not None:
                    l = cell.next_level_cells[:]
                    l[i] = new_next_level_cell
                    new_weight = cell.weight - old_next_level_cell_weight + new_next_level_cell.weight
                    self.Q[bag_num][q_input].push(Cell(cell.attribute2value, l, new_weight))
            # if q_input != "root" and self.Q[bag_num][q_input].size() > 0:
            if q_input != "root":
                # connect
                if self.Q[bag_num][q_input].size() > 0:
                    cell.next_cell = self.Q[bag_num][q_input].top()
        # else:                       # already connected
        #     if self.Q[bag_num][q_input].size() > 0 and cell.next_cell is self.Q[bag_num][q_input].top():
        #         return cell.next_cell
        #     else:
        #         return None

        if self.Q[bag_num][q_input].size() > 0:
            return self.Q[bag_num][q_input].top()
        else:
            return None

        # return cell.next_cell

    def create_q_input_topdown(self, cell, bag_num):
        child_q_input = []
        for key in self.bag[bag_num].keys:
            child_q_input += [key]
            child_q_input += ['=']
            child_q_input += [cell.attribute2value[key]]
            child_q_input += [',']
        child_q_input.pop()
        return "".join(str(x) for x in child_q_input)

if __name__ == "__main__":

    engine = RankEngine()

    engine.preprocess()
    print ("Q[0][root] top weight = " + str(engine.Q[0]["root"].top().weight))

    ret = engine.enumeration()
    print ("at 1th enumeration, the weight is " + str(ret.weight))
    
    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))

    ret = engine.enumeration()
    print ("at 2th enumeration, the weight is " + str(ret.weight))

    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))

    ret = engine.enumeration()
    print ("at 3th enumeration, the weight is " + str(ret.weight))

    ret = engine.enumeration()
    print ("at 4th enumeration, the weight is " + str(ret.weight))

    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print (str(engine.Q[1]["y=1"].heap[0].weight))
    print (str(engine.Q[1]["y=1"].heap[1].weight))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))
    print (str(engine.Q[3]["z=1"].heap[0].weight))

    ret = engine.enumeration()
    print ("at 5th enumeration, the weight is " + str(ret.weight))

    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print (str(engine.Q[1]["y=1"].heap[0].weight))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))

    ret = engine.enumeration()
    print ("at 6th enumeration, the weight is " + str(ret.weight))

    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))

    ret = engine.enumeration()
    print ("at 7th enumeration, the weight is " + str(ret.weight))

    ret = engine.enumeration()
    print ("at 8th enumeration, the weight is " + str(ret.weight))

    print ("Q[0][root] size=" + str(engine.Q[0]["root"].size()))
    print ("Q[1][y=1] size=" + str(engine.Q[1]["y=1"].size()))
    print ("Q[2][z=1] size=" + str(engine.Q[2]["z=1"].size()))
    print ("Q[3][z=1] size=" + str(engine.Q[3]["z=1"].size()))