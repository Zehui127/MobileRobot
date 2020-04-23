import heapq
import itertools

class Priority(object):
  def __init__(self):
    self._pq = []
    self._entry_finder = {}
    self._REMOVED = '<removed-task>'
    self._counter = itertools.count() 

  def add_task(self,task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in self._entry_finder:
        self.remove_task(task)
    count = next(self._counter)
    entry = [priority, count, task]
    self._entry_finder[task] = entry
    heapq.heappush(self._pq, entry)

  def remove_task(self,task):
      'Mark an existing task as REMOVED.  Raise KeyError if not found.'
      entry = self._entry_finder.pop(task)
      entry[-1] = self._REMOVED

  def pop_task(self):
      'Remove and return the lowest priority task. Raise KeyError if empty.'
      while self._pq:
          priority, count, task = heapq.heappop(self._pq)
          if task is not self._REMOVED:
              del self._entry_finder[task]
              return task
      raise KeyError('pop from an empty priority queue')







class PDict(object):
  def __init__(self,node):
    #self._min = node.cost
    #self._min_position = tuple(node.position)
    self._dict = {tuple(node.position):node}
    self._heap = Priority()
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