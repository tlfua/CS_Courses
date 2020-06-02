
class Relation:
    def __init__(self):
        self.attribute2index = {}
        self.tuple_set = set()
        self.keys = []

bag = {}

bag[0] = Relation()
bag[0].attribute2index['x'] = 0
bag[0].attribute2index['y'] = 1
bag[0].attribute2index['z'] = 2
bag[0].attribute2index['weight_0'] = 3
bag[0].tuple_set.add((1, 1, 1, 1))
bag[0].tuple_set.add((1, 2, 1, 1))
bag[0].tuple_set.add((1, 2, 2, 1))

bag[1] = Relation()
bag[1].attribute2index['w'] = 0
bag[1].attribute2index['x'] = 1
bag[1].attribute2index['y'] = 2
bag[1].attribute2index['weight_1'] = 3
bag[1].tuple_set.add((1, 1, 1, 1))
bag[1].tuple_set.add((1, 1, 2, 2))
bag[1].keys = ['x', 'y']

# key = {(0,1): ['x', 'y']}

traverse_order = [1, 0]

children = {0: [1]}

def preprocess():

    for bag_num in traverse_order:

        key_indexes = []
        for key in bag[bag_num].keys:
            key_indexes += [bag[bag_num].attribute2index[key]]
        
        for tuple_ in bag[bag_num].tuple_set:
            for index in key_indexes:
                print (str(tuple_[index]) + ' ', end='')
            print ()
            

            


if __name__ == "__main__":
    preprocess()
    
    