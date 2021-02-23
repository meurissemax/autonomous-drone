#!/usr/bin/env python

"""
Implementation of a tool to represent the environment
and interact with it.
"""

###########
# Imports #
###########

import math
import matplotlib.pyplot as plt
import numpy as np

from astar import AStar


############
# Settings #
############

plt.rcParams['toolbar'] = 'None'


###########
# Classes #
###########

class PathFinding(AStar):
    """
    Implementation of utility functions for the A* path
    planning algorithm.
    """

    def __init__(self, grid):
        self.grid = grid
        self.n, self.m = grid.shape

    def neighbors(self, node):
        i, j = node

        up = (i - 1, j)
        down = (i + 1, j)
        left = (i, j - 1)
        right = (i, j + 1)

        def accept(i, j):
            return 0 <= i < self.n and 0 <= j < self.m and self.grid[i, j] == 0

        return [(ni, nj) for ni, nj in [up, down, left, right] if accept(ni, nj)]

    def distance_between(self, n1, n2):
        return 1

    def heuristic_cost_estimate(self, n1, n2):
        (i1, j1) = n1
        (i2, j2) = n2

        return math.hypot(i2 - i1, j2 - j1)


class Environment:
    """
    Implementation of the representation of the
    environment.

    The agent is represented by its position and
    its orientation.
    """

    def __init__(self, env_pth):
        # Load the environment
        grid, lines = [], []
        pos, omega = [0, 0], 'N'
        obj = [0, 0]

        with open(env_pth, 'r') as env_file:
            lines = env_file.readlines()

        for i, line in enumerate(lines):
            row = []

            for j, char in enumerate(line):
                if char == '#':
                    row.append(1)
                elif char in ['.', '*', 'N', 'S', 'W', 'E']:
                    row.append(0)

                    if char == '*':
                        obj = [i, j]
                    elif char in ['N', 'S', 'W', 'E']:
                        pos, omega = [i, j], char

            grid.append(row)

        self.grid = np.array(grid)
        self.n, self.m = self.grid.shape

        # Agent
        self.pos, self.omega = np.array(pos), omega

        # Transition matrices
        self.t_move = {
            'forward': {'N': [-1, 0], 'S': [1, 0], 'W': [0, -1], 'E': [0, 1]},
            'left': {'N': 'W', 'S': 'E', 'W': 'S', 'E': 'N'},
            'right': {'N': 'E', 'S': 'W', 'W': 'N', 'E': 'S'},
        }

        self.t_actions = {
            'up': {'N': 'forward', 'S': 'right', 'W': 'right', 'E': 'left'},
            'down': {'N': 'right', 'S': 'forward', 'W': 'left', 'E': 'right'},
            'left': {'N': 'left', 'S': 'right', 'W': 'forward', 'E': 'right'},
            'right': {'N': 'right', 'S': 'left', 'W': 'right', 'E': 'forward'}
        }

        # Objective
        self.obj = np.array(obj)

        # Path finder
        self.path_finder = PathFinding(self.grid)

        # Plot
        self.xticks = np.arange(0, self.m, 1)
        self.yticks = np.arange(0, self.n, 1)

        self.markers = {
            'N': '^',
            'S': 'v',
            'W': '<',
            'E': '>'
        }

    def has_reached_obj(self):
        return (self.pos == self.obj).all()

    def _next(self, pos, omega, d):
        transition = self.t_move.get(d).get(omega)

        # Move
        if d == 'forward':
            pos = pos + np.array(transition)
            pos = np.array([np.maximum(0, pos[0]), np.minimum(pos[1], self.m)])

        # Rotate
        elif d in ['left', 'right']:
            omega = transition

        return pos, omega

    def move(self, d, times=1):
        for _ in range(times):
            self.pos, self.omega = self._next(self.pos, self.omega, d)

    def update(self, pos=None, omega=None, obj=None):
        def accept(i, j):
            return 0 <= i < self.n and 0 <= j < self.m and self.grid[i, j] == 0

        # Position
        if pos is not None:
            i, j = pos

            if accept(i, j):
                self.pos = np.array(pos)

        # Orientation
        if omega is not None and omega in ['N', 'S', 'W', 'E']:
            self.omega = omega

        # Objective
        if obj is not None:
            i, j = obj

            if accept(i, j):
                self.obj = np.array(obj)

    def path(self, start=None, end=None):
        a = start if start is not None else tuple(self.pos)
        b = end if end is not None else tuple(self.obj)

        path = np.array(list(self.path_finder.astar(a, b)))

        return path[1:]

    def sequence(self, path):
        # Initial position and orientation
        pos, omega = self.pos, self.omega

        # Actions
        actions = []

        # Index of current point
        idx = 0
        n_points = path.shape[0]

        # Iterate over each point of the path
        while idx < n_points:
            point = path[idx]

            i, j = pos[0], pos[1]

            # Get neighbors of the current position
            neighbors = {
                'up': [i - 1, j],
                'down': [i + 1, j],
                'left': [i, j - 1],
                'right': [i, j + 1]
            }

            # Check which neighbor the current point is
            neighbor = None

            for key, value in neighbors.items():
                if (point == value).all():
                    neighbor = key

            # If next point is a valid neighbor
            if neighbor is not None:

                # Get action to this neighbor
                action = self.t_actions.get(neighbor).get(omega)
                actions.append(action)

                idx = idx + 1 if action == 'forward' else idx

                # Update theoretical position and orientation
                pos, omega = self._next(pos, omega, action)
            else:
                break

        return actions

    def group(self, actions, limit=5):
        grouped = []
        previous = None

        for action in actions:
            add = not(action == previous)

            if action == previous:
                if action == 'forward' and grouped[-1][1] >= limit:
                    add = True
                else:
                    grouped[-1][1] += 1

            if add:
                grouped.append([action, 1])

            previous = action

        grouped = [tuple(action) for action in grouped]

        return grouped

    def show(self, pos=False, obj=False, points=None):
        # Grid
        plt.figure('Environment')
        plt.clf()
        plt.imshow(self.grid, cmap='Greys')

        plt.xticks(self.xticks)
        plt.yticks(self.yticks)

        # Points (if any)
        if points is not None:
            points = np.array(points)
            i, j = points[:, 1], points[:, 0]

            plt.scatter(i, j, c='blue')

        # Agent position
        if pos:
            i, j = self.pos[1], self.pos[0]
            marker = self.markers.get(self.omega)

            plt.scatter(i, j, c='red', marker=marker)

        # Objective
        if obj:
            i, j = self.obj[1], self.obj[0]

            plt.scatter(i, j, c='yellow', marker='*')

        # Axis parameters
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.xaxis.tick_top()

        # Show
        plt.draw()
        plt.pause(0.5)

    def keep(self):
        plt.show()


########
# Main #
########

def main(
    env_pth='indoor-corridor.txt'
):
    # Create the environment
    env = Environment(env_pth)

    # Compute shortest path to objective
    path = env.path()

    # Get sequence of actions
    sequence = env.sequence(path)

    print('Sequence of actions')
    print(sequence)

    # Group sequence of action
    grouped = env.group(sequence)

    print('Grouped sequence of actions')
    print(grouped)

    # Has the drone reached its objective ?
    print(f'Has reached objective ? {env.has_reached_obj()}')

    # Show the environment
    env.show(pos=True, obj=True, points=path)
    env.keep()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Environment representation.')

    parser.add_argument('-e', '--environment', type=str, default='indoor-corridor.txt', help='path to environment file')

    args = parser.parse_args()

    main(
        env_pth=args.environment
    )
