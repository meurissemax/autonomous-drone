"""
Implementation of tools to represent an environment and interact with it.
"""

###########
# Imports #
###########

import math
import numpy as np
import operator

from astar import AStar
from typing import Dict, List, Tuple

from plots.latex import plt


##########
# Typing #
##########

Grid = np.array

Position = Tuple[int, int]
Orientation = str  # N, S, W, E

Action = str  # forward, left, right

Objective = Position
Path = List[Position]

Keypoint = Tuple[Action, Position]

Batteries = List[Position]


###########
# Classes #
###########

class PathFinding(AStar):
    """
    Implementation of utility functions for the A* path planning algorithm.
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
    Implementation of the representation of the environment.

    An environment representation must be stored in a text file (.txt). Its
    shape has to be a grid (m * n elements).

    Features of the environment are represented by a serie of characters, i.e.
        . : a free position
        # : a non free position
        [N, S, W, E] : position of the agent, represented by a letter
                       indicating its orientation
        B : a battery station
        * : the objective to reach (only once by representation)
        + : a staircase that goes up THAT THE DRONE MUST TAKE. Hence,
            this character acts as an objective. If there is a staircase
            but the drone don't need to take it, no need to use this
            character (so, only once by representation, if needed)
        - : a staircase that goes down (same remark for the staircase)

    Example of a small environment:

    ######
    #*...#
    #...B#
    #....#
    #...N#
    ######
    """

    # Initialization

    def __init__(self, env_pth: str):
        # Actions
        self.move_actions = ['forward']
        self.turn_actions = ['left', 'right']
        self.orientations = ['N', 'S', 'W', 'E']

        # Transition matrices
        self.t_move = {
            'forward': {'N': [-1, 0], 'S': [1, 0], 'W': [0, -1], 'E': [0, 1]},
            'left': {'N': 'W', 'S': 'E', 'W': 'S', 'E': 'N'},
            'right': {'N': 'E', 'S': 'W', 'W': 'N', 'E': 'S'},
            'up': {'N': 'N', 'S': 'S', 'W': 'W', 'E': 'E'},
            'down': {'N': 'N', 'S': 'S', 'W': 'W', 'E': 'E'}
        }

        self.t_actions = {
            'up': {'N': 'forward', 'S': 'right', 'W': 'right', 'E': 'left'},
            'down': {'N': 'right', 'S': 'forward', 'W': 'left', 'E': 'right'},
            'left': {'N': 'left', 'S': 'right', 'W': 'forward', 'E': 'right'},
            'right': {'N': 'right', 'S': 'left', 'W': 'right', 'E': 'forward'}
        }

        # Plot
        self.markers = {
            'N': '^',
            'S': 'v',
            'W': '<',
            'E': '>'
        }

        # Battery
        self.full_battery_distance = 200

        # Load the environment and everything related
        self.load(env_pth)

    def load(self, env_pth: str):
        """
        Load an environment representation and save all information about it.
        """

        # Initialize
        grid, lines = [], []
        pos, omega = None, None
        obj = None
        staircases = []
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
                elif char == '.':
                    row.append(0)
                elif char in self.orientations:
                    row.append(0)

                    pos, omega = (i, j), char
                elif char == 'B':
                    row.append(0)

                    batteries.append((i, j))
                elif char in ['*', '+', '-']:
                    row.append(0)

                    obj = (i, j)

                    if char == '+':
                        staircases.append(('up', (i, j)))
                    elif char == '-':
                        staircases.append(('down', (i, j)))

            grid.append(row)

        # Save information
        self.grid = np.array(grid)
        self.n, self.m = self.grid.shape
        self.pos, self.omega = pos, omega
        self.obj = obj
        self.staircases = staircases
        self.batteries = batteries

        # Path finder
        self.path_finder = PathFinding(self.grid)

        # Plot
        self.xticks = np.arange(0, self.m, 1)
        self.yticks = np.arange(0, self.n, 1)

    # Misc

    def _bound(self, p: Position) -> Position:
        """
        Bound a position in the environment limits.
        """

        i, j = p

        i = min(max(0, i), self.n - 1)
        j = min(max(0, j), self.m - 1)

        return i, j

    def _is_free(self, p: Position) -> bool:
        """
        Check if a position is valid and free.
        """

        i, j = p

        return 0 <= i < self.n and 0 <= j < self.m and self.grid[i, j] == 0

    def _neighbors(self, p: Position) -> Dict[str, Position]:
        """
        Get neighbors of a position.
        """

        i, j = p

        neighbors = {
            'up': (i - 1, j),
            'down': (i + 1, j),
            'left': (i, j - 1),
            'right': (i, j + 1)
        }

        return neighbors

    # Battery

    def _battery_distance(self, battery: int) -> int:
        """
        Get the distance (in meters) the drone can navigate with a given
        battery level.
        """

        return math.floor(self.full_battery_distance * (battery / 100))

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

    # Move in environment

    def _next(
        self,
        pos: Position,
        omega: Orientation,
        action: Action
    ) -> Tuple[Position, Orientation]:
        """
        Get the new position and orientation of the agent if it executes a
        certain action.
        """

        transition = self.t_move.get(action).get(omega)

        # Move
        if action in self.move_actions:
            npos = tuple(i + j for i, j in zip(pos, transition))
            npos = self._bound(npos)

            pos = npos if self._is_free(npos) else pos

        # Rotate
        elif action in self.turn_actions:
            omega = transition

        return pos, omega

    def move(self, action: Action, times: int = 1):
        """
        Update the position and orientation of the agent based on an action.
        """

        for _ in range(times):
            self.pos, self.omega = self._next(self.pos, self.omega, action)

    def update(self, p: Position, action: Action):
        """
        Update the position and orientation of the agent based on new position
        and an action.
        """

        # Update position
        if self._is_free(p):
            self.pos = p

        # Update orientation
        if action in self.turn_actions:
            _, self.omega = self._next(self.pos, self.omega, action)

    # Path planning and corresponding actions

    def path(
        self,
        start: tuple = None,
        end: tuple = None,
        battery: int = None,
        history: list = []
    ) -> Path:
        """
        Get the shortest path, represented by a serie of positions, to go from
        a start point to an end point.

        The path can take into account a battery level if the latter is given
        as argument.
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
        Translate a path to the corresponding sequence of actions, e.g. move
        forward, move forward, turn left, move forward, etc.
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

            # Get neighbors of the current position
            neighbors = self._neighbors(pos)

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

                idx = idx + 1 if action in self.move_actions else idx

                # Update theoretical position and orientation
                pos, omega = self._next(pos, omega, action)
            else:
                break

        # Check if objective is a staircase
        obj = path[-1]

        for action, p in self.staircases:
            if obj == p:
                actions.append(action)

        return actions

    def group_seq(
        self,
        actions: List[Action],
        limit: int = 5
    ) -> List[Tuple[Action, int]]:
        """
        Group a serie of actions into small group of same actions in order to
        optimize drone navigation, e.g. move forward, move forward will be
        grouped into (move forward, 2).

        A limited number of same consecutive actions can be grouped into a
        single group.
        """

        grouped = []
        previous = None

        for action in actions:
            add = not(action == previous)

            if action == previous:
                if action in self.move_actions and grouped[-1][1] >= limit:
                    add = True
                else:
                    grouped[-1][1] += 1

            if add:
                grouped.append([action, 1])

            previous = action

        grouped = [tuple(action) for action in grouped]

        return grouped

    # Key points

    def _is_keypoint(self, p: Position) -> bool:
        """
        Check if a point of the environment is a key point.
        """

        # Get neighbors of the point
        neighbors = list(self._neighbors(p).values())

        # Filter by free neighbors
        free = [n for n in neighbors if self._is_free(n)]

        # Using free neighbors, check if point is a key point
        n_unique = 0

        for axis in [0, 1]:
            n = len(set([p[axis] for p in free]))
            n_unique += n

        return n_unique > 3

    def extract_keypoints(
        self,
        path: Path,
        actions: List[Action]
    ) -> List[Keypoint]:
        """
        Isolate, from a path and its corresponding serie of actions, the key
        points and their corresponding positions.
        """

        # Get key point positions of the path
        points = [p for p in path if self._is_keypoint(p)]

        # Get type of each key point
        types = []

        previous = None
        pos, omega = self.pos, self.omega

        for action in actions:
            if pos in points and pos != previous:
                types.append(action)

            previous = pos
            pos, omega = self._next(pos, omega, action)

        # Construct list of key points
        keypoints = list(zip(types, points))

        return keypoints

    # Objective

    def has_reached_obj(self) -> bool:
        """
        Check if the agent has reached its objective.
        """

        return self.pos == self.obj

    # Reinforcement Learning utilities

    def to_rewards(self) -> Grid:
        """
        Create the rewarded version of the environment.
        """

        rewards = self.grid.copy()

        # Free positions
        rewards = np.where(self.grid == 0, -1, rewards)

        # Non free positions
        rewards = np.where(self.grid == 1, -1e10, rewards)

        # Objective
        rewards[self.obj] = 100

        return rewards

    # Rendering

    def render(
        self,
        draw: bool = True,
        export: str = None,
        path: Path = None,
        what: list = []
    ):
        """
        Generates a rendering of the environment. It can then be shown or
        saved.
        """

        # Grid
        plt.figure('Environment', figsize=(8, 6))
        plt.clf()

        plt.grid(False)

        plt.imshow(self.grid, cmap='Greys')

        plt.xticks(self.xticks)
        plt.yticks(self.yticks)

        # Batteries
        if 'bat' in what:
            for b in self.batteries:
                j, i = b

                plt.scatter(i, j, c='green', marker='P', s=100)

        # Objective
        if 'obj' in what and self.obj is not None:
            j, i = self.obj

            plt.scatter(i, j, c='orange', marker='*', s=100)

        # Agent position
        if 'pos' in what and self.pos is not None:
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
        Keep the environment representation opened when showing it. Must be
        called once at the end of a script that shows the environment.
        """

        plt.show()
