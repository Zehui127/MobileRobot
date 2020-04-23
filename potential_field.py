from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_RADIUS = .2
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5

# EXTRA Obstacles
O1 =  np.array([0, 0.5], dtype=np.float32)
O2 =  np.array([0.5, 0], dtype=np.float32)
R1 = .2
R2 = .2

def dist(a,b):
  return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def get_velocity_to_reach_goal(position, goal_position):
  v = np.zeros(2, dtype=np.float32)
  # MISSING: Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.
  if not (position[0] == goal_position[0] and position[1] == goal_position[1]):
    v = np.random.normal(0,0.05,size=2)
  d_threshold = 1
  force_factor = 1
  d = dist(position,goal_position)
  if d <= d_threshold:
    return v[0]-force_factor*(position[0]-goal_position[0]),v[1]-force_factor*(position[1]-goal_position[1])
  else:
    return v[0]-d_threshold*force_factor*(position[0]-goal_position[0])/dist(position,goal_position),v[1]-d_threshold*force_factor*(position[1]-goal_position[1])/dist(position,goal_position)


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  v = np.zeros(2, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid the obstacles
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). Make sure to not create
  # speeds that are larger than max_speed for each obstacle. Both obstacle_positions
  # and obstacle_radii are lists.
  q_threshold = 1.5
  force_factor = 1
  for i in range(len(obstacle_positions)):
    pos = obstacle_positions[i]
    rad = obstacle_radii[i]
    dist_obst = abs(dist(position,pos)-rad)
    if dist_obst <= q_threshold:
      v[0] += -force_factor*(1/q_threshold - 1/dist_obst)*1/(dist_obst**2)*(position[0]-pos[0])/dist_obst
      v[1] += -force_factor*(1/q_threshold - 1/dist_obst)*1/(dist_obst**2)*(position[1]-pos[1])/dist_obst
  # Solve the problem with a rule based solution
  #if position[0] == position[1]:
  #  v[0] += 1
  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      [CYLINDER_POSITION],
      [CYLINDER_RADIUS])
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  # add extra obstacles
  #ax.add_artist(plt.Circle(O1, R1, color='gray'))
  #ax.add_artist(plt.Circle(O2, R2, color='gray'))
  #####
  ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()
