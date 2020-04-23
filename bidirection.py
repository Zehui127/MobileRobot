from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
from PDict import PDict
import time
np.random.seed(1000)

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.



FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500


def sample_random_position(occupancy_grid):
  position = np.zeros(2, dtype=np.float32)
  position = np.random.uniform(low=-2,high=2,size=2)
  while not occupancy_grid.is_free(position):
    position = np.random.uniform(low=-2,high=2,size=2)

  # MISSING: Sample a valid random position (do not sample the yaw).
  # The corresponding cell must be free in the occupancy grid.
  return position


def adjust_pose(node, final_position, occupancy_grid):
  def collision_free(start,end,occupancy_grid,n=10):
    #sample 10 points from the arc to check if there are obstacles on the arc
    center,radius = find_circle(start,end)
    d_start = start.position - center
    theta1 = np.arctan2(d_start[1],d_start[0])
    d_end = end.position - center
    theta2 = np.arctan2(d_end[1],d_end[0])
    diff = theta2 - theta1
    # Check if the arc goes clockwise.
    if np.cross(start.direction, d_start).item() > 0.:
      theta1, theta2 = theta2, theta1
      diff = np.pi*2 + theta2 - theta1
    for frac_angle in np.linspace(0,abs(diff),n):
      temp_position = np.zeros(2)
      theta = theta1 + frac_angle
      temp_position[0] = radius*np.cos(theta) + center[0] 
      temp_position[1] = radius*np.sin(theta) + center[1]
      if not occupancy_grid.is_free(temp_position):
        return False
    return True
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  #del_theta = np.arctan2(final_position[1],final_position[0]) - np.arctan2(node.pose[1],node.pose[0])
  temp_a = (node.pose[Y]-final_pose[Y],final_pose[X]-node.pose[X])
  temp_b = (-node.direction[1],node.direction[0])
  cos_theta = np.dot(temp_a,temp_b)/(np.linalg.norm(temp_b)*np.linalg.norm(temp_a))
  del_theta = np.arccos(cos_theta)*2
  # decide the rotational direction
  center,_ = find_circle2(node,Node(final_pose))
  d_start = node.position - center
  # Check if the arc goes clockwise.
  if np.cross(node.direction, d_start).item() > 0.:
    final_pose[2] = node.yaw - del_theta
  else:
    final_pose[2] = node.yaw + del_theta
  if collision_free(node,Node(final_pose),occupancy_grid):
    return Node(final_pose)
  # generate YAW set
  # check if there exists an obstacle on the arc
  # MISSING: Check whether there exists a simple path that links node.pose
  # to final_position. This function needs to return a new node that has
  # the same position as final_position and a valid yaw. The yaw is such that
  # there exists an arc of a circle that passes through node.pose and the
  # adjusted final pose. If no such arc exists (e.g., collision) return None.
  # Assume that the robot always goes forward.
  # Feel free to use the find_circle() function below.
  return None


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def remove_neighbor(self,node):
    self._neighbors.remove(node)
  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  @yaw.setter
  def yaw(self, c):
    self._pose[YAW] = c
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c
def Euclidean(position,des_position):
  """Return the Euclidean Distance Heuristic to estimate the distance between current node and the goal node"""
  return np.linalg.norm(position-des_position)

def addNode(queue,node,dict_obj):
  heapq.heappush(queue,(node.cost,node))
  dict_obj[node.position] = node
  #addNode(queue,middle)
def pop(queue,dict_obj):
  # average log(n)
  node = heapq.heappop(queue)[1]
  del dict_obj[node.position]
  return node
  #pop(queue)
def generate_neighbour(node,map):
  GRID = 0.2
  result = []
  count = 0
  for i in [-1,0,1]:
    for j in  [-1,0,1]:
      if i == 0 and j ==0:
        continue
      new_position  = node.position + GRID*np.array([i,j])
      new_position = np.around(new_position, decimals=2)
      if map.is_free(new_position): # within the boundary of the map
        new_node = adjust_pose(node,new_position,map)
        if new_node:
          node.add_neighbor(new_node)
          result.append(new_node)
  return result
  """Generate the neighbour nodes of the current node in 4 direction"""
  # it dependents on the shape of the map, now we assume we are in a -2 to 2, i.e. 4*4 square
  # use 0.1 as the length of the grid 40*40 grid (1600) grid in total

def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = [] 
  start_node2 = Node(np.array([goal_position[0],goal_position[1],0.],dtype=np.float32))
  start_node2.cost = 0
  final_node2 = None
  open_list2 = PDict(start_node2)
  #------------------------------------------------------
  start_node = Node(start_pose)
  # set stat.f = 0
  start_node.cost = 0
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  open_list = PDict(start_node)
  closed_list = set([])

  common_list = PDict(start_node)
  common_list2 = PDict(start_node2)


  while not open_list.empty():
    current_node2  = open_list2.pop()
    if tuple(current_node2.position) not in closed_list:
      closed_list.add(tuple(current_node2.position))
    else:
      print(2)
      return start_node, common_list.getNode(current_node2.position),start_node2,common_list2.getNode(current_node2.position)

    # find the element with least f in the open list and pop
    current_node = open_list.pop()
    if tuple(current_node.position) not in closed_list:
      closed_list.add(tuple(current_node.position))
    else:
      print(1)
      return start_node, common_list.getNode(current_node.position),start_node2,common_list2.getNode(current_node.position)
       # generate 8 successor and 
    # set their parents to curent node
    node_list2 = generate_neighbour(current_node2,occupancy_grid)
    for ele in node_list2:
      #if successor is the goal, return 
      if Euclidean(ele.position,start_node.position) < .2:
        ele.parent = current_node2
        final_node = ele
        print("return here2")
        return start_node2, final_node
      if tuple(ele.position) in closed_list:
        continue
      cost_g = current_node2.cost  + Euclidean(ele.position,current_node2.position)
      cost_h = Euclidean(ele.position,start_node.position)
      ele.cost = cost_g + cost_h
      ele.parent = current_node2
      #if node is not in the open list (from position pof) or if the current cost smaller than cost of the node withe the same position 
      if not open_list2.is_in(ele.position):
        open_list2.addNode(ele)
        common_list2.addNode(ele)
      elif ele.cost< open_list2.getNode(ele.position).cost:
        open_list2.updateNode(ele)

    # generate 8 successor and 
    # set their parents to curent node
    node_list = generate_neighbour(current_node,occupancy_grid)
    for ele in node_list:
      #if successor is the goal, return 
      if Euclidean(ele.position,goal_position) < .2:
        ele.parent = current_node
        final_node = ele
        print("return here")
        #return start_node, final_node
      if tuple(ele.position) in closed_list:
        continue
      cost_g = current_node.cost  + Euclidean(ele.position,current_node.position)
      cost_h = Euclidean(ele.position,goal_position)
      ele.cost = cost_g + cost_h
      ele.parent = current_node
      #if node is not in the open list (from position pof) or if the current cost smaller than cost of the node withe the same position 
      if not open_list.is_in(ele.position):
        open_list.addNode(ele)
        common_list.addNode(ele)
      elif ele.cost< open_list.getNode(ele.position).cost:
        open_list.updateNode(ele)
  return start_node, final_node

def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)

def find_circle2(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_a.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_a.position
  return center, np.abs(radius)



def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    #plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
    #          head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    #plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
    #          head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    #ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
    #                         theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
    #                         color=color, lw=lw))
    line = plt.Line2D(tuple([u.position[0],v.position[0]]), tuple([u.position[1],v.position[1]]), color=color,lw=lw)
    ax.add_line(line)

  points = []
  s = [(start_node, None)]  # (node, parent).
  num_path = 0
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      num_path += 1
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))
  print("************NumberOfPath: ",num_path)
  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      print(v.pose[0],v.pose[1],v.yaw,"v")
      print(v.parent.pose[0],v.parent.pose[1],v.parent.yaw,"parent")
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  start_time = time.time()
  start_node, final_node,start_node2,final_node2 = rrt(START_POSE, GOAL_POSITION, occupancy_grid)
  print("--- %s seconds ---" % (time.time() - start_time))
  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  draw_solution(start_node2,final_node2)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)
  
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 2., 2. + .5])
  plt.ylim([-.5 - 2., 2. + .5])
  plt.show()
#Dynamic weighting - set the wieght param to 0.8 imcrease the performance
#Bidirectional search -
#Iterative deepening
#Jump Point Search -