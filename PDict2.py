import heapq
import itertools
import bisect
import sys
class Priority(object):
  def __init__(self,size):
    self._pq = []
    self._REMOVED = float("inf")
    self._size = size

  def add_task(self,task, priority=0):
    def find_entry(array,target):
      for ind,ele in enumerate(array):
        if ele == target:
          return ind
    'Add a new task or update the priority of an existing task'
    for ind,(p,ele) in enumerate(self._pq):
      if ele == task:
        del self._pq[ind]
    bisect.insort_right(self._pq, (priority, task))
    self._pq = self._pq[:self._size]

  def pop_task(self):
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    temp = self._pq[0][1]
    del self._pq[0]
    return temp






class PDict(object):
  def __init__(self,node):
    #self._min = node.cost
    #self._min_position = tuple(node.position)
    self._dict = {tuple(node.position):node}
    self._heap = Priority(10)
    self._heap.add_task(tuple(node.position),node.cost)
    
  def addNode(self,node):
    self._dict[tuple(node.position)] = node
    self._heap.add_task(tuple(node.position),node.cost)
  def pop(self):
    position = self._heap.pop_task()
    node = self._dict[position]
    del self._dict[position]
    return node

  def getNode(self,position):
    return self._dict[tuple(position)]

  def updateNode(self,node):
    position = node.position
    cost = node.cost
    position = tuple(position)
    self._dict[position].cost = cost
    self._heap.add_task(position,cost)
    self._dict[position].parent = node.parent 
    #self._dict[position].neighbors = node.neighbors
  def is_in(self,position):
    return tuple(position) in self._dict
  def empty(self):
    if self._dict:
      return False
    else:
      return True