#!/usr/bin/env python

"""
Implementation of a tool to represent the environment
and interact with it.
"""

###########
# Imports #
###########

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import operator

from astar import AStar
from typing import List, Tuple


##########
# Typing #
##########

Grid = np.array

Position = Tuple[int, int]
Orientation = str  # N, S, W, E

Direction = str  # up, down, left, right
Action = str  # forward, left, right

Objective = Position
Path = List[Position]

Batteries = List[Position]


############
# Settings #
############

plt.rcParams['toolbar'] = 'None'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True
plt.rcParams['savefig.transparent'] = True

if mpl.checkdep_usetex(True):
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True


#####################
# General variables #
#####################

# Number of meters the drone can cover with a fully charged battery.
FULL_BATTERY_DISTANCE = 200


###########
# Classes #
###########

class PathFinding(AStar):
    """
    Implementation of utility functions for the A* path planning
    algorithm.
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

        neighbors = [up, down, left, right]

        def accept(i, j):
            return 0 <= i < self.n and 0 <= j < self.m and self.grid[i, j] == 0

        return [(ni, nj) for ni, nj in neighbors if accept(ni, nj)]

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

    def __init__(self, env_pth: str):
        # Load the environment
        grid, pos, omega, obj, batteries = self._load_env(env_pth)

        self.grid = grid
        self.n, self.m = grid.shape
        self.pos, self.omega = pos, omega
        self.obj = obj
        self.batteries = batteries

        # Path finder
        self.path_finder = PathFinding(self.grid)

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

        # Plot
        self.xticks = np.arange(0, self.m, 1)
        self.yticks = np.arange(0, self.n, 1)

        self.markers = {
            'N': '^',
            'S': 'v',
            'W': '<',
            'E': '>'
        }

    def _load_env(
        self,
        env_pth: str
    ) -> Tuple[Grid, Position, Orientation, Objective, Batteries]:
        """
        Load the environment stored in a text file and get all information
        about it.
        """

        # Initialize
        grid, lines = [], []
        pos, omega = (0, 0), 'N'
        obj = (0, 0)
        batteries = []

        # Get lines of the file
        with open(env_pth, 'r') as env_file:
            lines = env_file.readlines()

        # Create the environment
        for i, line in enumerate(lines):
            row = []

            for j, char in enumerate(line):
                if char == '#':
                    row.append(1)
                elif char in ['.', '*', 'B', 'N', 'S', 'W', 'E']:
                    row.append(0)

                    if char == '*':
                        obj = (i, j)
                    elif char == 'B':
                        batteries.append((i, j))
                    elif char in ['N', 'S', 'W', 'E']:
                        pos, omega = (i, j), char

            grid.append(row)

        # Convert
        grid = np.array(grid)

        return grid, pos, omega, obj, batteries

    def _battery_distance(self, battery: int) -> int:
        """
        Get the distance (in meters) the drone can navigate with a given
        battery level.
        """

        return math.floor(FULL_BATTERY_DISTANCE * (battery / 100))

    def _farthest_battery(self, battery: int) -> tuple:
        """
        Get the farthest reachable battery station, if any.
        """

        # If no battery station
        if len(self.batteries) == 0:
            return None

        # Get battery station and their distance
        d_batteries = []

        for b in self.batteries:
            path = self.path(end=b)

            d_batteries.append((
                len(path),
                b
            ))

        # Sort by distances
        d_batteries.sort(key=operator.itemgetter(0), reverse=True)

        # Distance the drone can navigate
        distance = self._battery_distance(battery)

        # Get the farthest reachable battery station
        for d, b in d_batteries:
            if d <= distance:
                return b

        return None

    def _next(
        self,
        pos: Position,
        omega: Orientation,
        d: Direction
    ) -> Tuple[Position, Orientation]:
        """
        Get the new position and orientation of the agent if it moves in a
        certain direction.
        """

        transition = self.t_move.get(d).get(omega)

        # Move
        if d == 'forward':
            pos = tuple(i + j for i, j in zip(pos, transition))

            pos = max(0, pos[0]), max(0, pos[1])
            pos = min(pos[0], self.n - 1), min(pos[1], self.m - 1)

        # Rotate
        elif d in ['left', 'right']:
            omega = transition

        return pos, omega

    def move(self, d: Direction, times: int = 1):
        """
        Effectively move the agent in the environment
        representation.
        """

        for _ in range(times):
            self.pos, self.omega = self._next(self.pos, self.omega, d)

    def path(
        self,
        start: tuple = None,
        end: tuple = None,
        battery: int = None,
        history: list = []
    ) -> Path:
        """
        Get the shortest path, represented by a serie of positions, to go
        from a start point to an end point.

        The path can take into account a battery level if the latter
        is given as argument.
        """

        a = start if start is not None else self.pos
        b = end if end is not None else self.obj

        path = list(self.path_finder.astar(a, b))
        path = path[1:]

        # If battery level is given
        if battery is not None:

            # Distance the drone can navigate
            distance = self._battery_distance(battery)

            # Distance to objective
            d_obj = len(path)

            # If the drone can not reach the objective
            if distance < d_obj:
                nearest = self._farthest_battery(battery=battery)

                # If the drone can reach a battery station
                if nearest is not None and nearest not in history:
                    path = self.path(end=nearest) + self.path(
                        start=nearest,
                        end=b,
                        battery=100,
                        history=history.append(nearest)
                    )
                else:
                    path = []

        return path

    def path_to_seq(self, path: Path) -> List[Action]:
        """
        Translate a path to the corresponding sequence of actions, e.g.
        move forward, move forward, turn left, move forward, etc.
        """

        # Initial position and orientation
        pos, omega = self.pos, self.omega

        # Actions
        actions = []

        # Index of current point
        idx = 0
        n_points = len(path)

        # Iterate over each point of the path
        while idx < n_points:
            point = path[idx]

            i, j = pos

            # Get neighbors of the current position
            neighbors = {
                'up': (i - 1, j),
                'down': (i + 1, j),
                'left': (i, j - 1),
                'right': (i, j + 1)
            }

            # Check which neighbor the current point is
            neighbor = None

            for key, value in neighbors.items():
                if point == value:
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

    def isolate_turns(
        self,
        actions: List[Action]
    ) -> List[Action]:
        """
        Isolate, from a serie of actions, the turning actions.
        """

        turn_actions = ['left', 'right']

        return [a for a in actions if a in turn_actions]

    def group_seq(
        self,
        actions: List[Action],
        limit: int = 5
    ) -> List[Tuple[Action, int]]:
        """
        Group a serie of actions into small group of same actions in order
        to optimize drone navigation, e.g. move forward, move forward will
        be grouped into (move forward, 2).

        A limited number of same consecutive actions can be grouped into a
        single group.
        """

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

    def has_reached_obj(self) -> bool:
        """
        Check if the agent has reached its objective.
        """

        return self.pos == self.obj

    def render(
        self,
        draw: bool = True,
        export: str = None,
        path: Path = None,
        what: list = []
    ):
        """
        Generates a rendering of the environment. It can then be shown
        or saved.
        """

        # Grid
        plt.figure('Environment', figsize=(8, 6))
        plt.clf()
        plt.imshow(self.grid, cmap='Greys')

        plt.xticks(self.xticks)
        plt.yticks(self.yticks)

        # Batteries
        if 'bat' in what:
            for b in self.batteries:
                j, i = b

                plt.scatter(i, j, c='green', marker='P', s=100)

        # Objective
        if 'obj' in what:
            j, i = self.obj

            plt.scatter(i, j, c='orange', marker='*', s=100)

        # Agent position
        if 'pos' in what:
            j, i = self.pos
            marker = self.markers.get(self.omega)

            plt.scatter(i, j, c='red', marker=marker, s=100)

        # Path
        if path is not None and len(path) > 0:
            path = np.array(path)
            i, j = path[:, 1], path[:, 0]

            plt.scatter(i, j, c='blue', alpha=0.25)

        # Axis parameters
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.xaxis.tick_top()

        # Export
        if export is not None:
            plt.savefig(export)

        # Draw
        if draw:
            plt.draw()
            plt.pause(0.5)

    def keep(self):
        """
        Keep the environment representation opened when showing it. Must
        be called once at the end of a script that shows the environment.
        """

        plt.show()


########
# Main #
########

def main(
    env_pth: str = 'indoor-corridor.txt',
    battery: int = None
):
    # Create the environment
    env = Environment(env_pth)

    # Compute shortest path to objective
    path = env.path(battery=battery)

    # Get sequence of actions
    sequence = env.path_to_seq(path)

    print('Sequence of actions')
    print(sequence)

    # Isolate turn actions
    turns = env.isolate_turns(sequence)

    print('Turn actions')
    print(turns)

    # Group sequence of action
    grouped = env.group_seq(sequence)

    print('Grouped sequence of actions')
    print(grouped)

    # Has the drone reached its objective ?
    print(f'Has reached objective ? {env.has_reached_obj()}')

    # Show the environment
    env.render(path=path, what=['pos', 'obj', 'bat'])
    env.keep()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Environment representation.'
    )

    parser.add_argument(
        '-e',
        '--environment',
        type=str,
        default='indoor-corridor.txt',
        help='path to environment file'
    )

    parser.add_argument(
        '-b',
        '--battery',
        type=int,
        default=None,
        help='battery level of the drone'
    )

    args = parser.parse_args()

    main(
        env_pth=args.environment,
        battery=args.battery
    )
