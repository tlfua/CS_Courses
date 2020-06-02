
from heapq import heappush, heappop

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

if __name__ == "__main__":
    q = PriorityQueue()
    q.push(3)
    q.push(1)
    q.push(5)

    while q.size() > 0:
        print (q.pop())