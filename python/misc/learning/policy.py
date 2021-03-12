#!/usr/bin/env python

"""
Implementation of a very simple algorithm to determine the best policy to reach
objective.

This implementation is inspired from a project I realized with François Rozet:
    - https://github.com/francois-rozet/info8003.git
"""

###########
# Imports #
###########

import itertools
import math
import numpy as np
import os
import sys

from tqdm import tqdm
from typing import List, Tuple

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from uav.environment import Environment  # noqa: E402
from plots.latex import plt  # noqa: E402


##########
# Typing #
##########

Grid = np.array
Shape = Tuple[int, int]

State = Tuple[int, int]
Action = Tuple[int, int]
Reward = float


#############
# Functions #
#############

def dynamics(x: State, u: Action, shape: Shape) -> State:
    """
    Dynamics of the system. Get the new state when agent executes action 'u'
    from state 'x'.
    """

    x, y = x
    i, j = u

    n, m = shape

    return min(max(x + i, 0), n - 1), min(max(y + j, 0), m - 1)


def reward(x: State, u: Action, g: Grid) -> Reward:
    """
    Reward obtained by executing action 'u' from state 'x'.
    """

    return g[dynamics(x, u, g.shape)]


def mdp(g: Grid, U: List[Action]) -> Tuple[np.array, np.array]:
    """
    Markov Decision Process mappings.
    """

    n, m = g.shape

    er = np.zeros((len(U), n, m))  # expected reward
    tp = np.zeros((len(U), n, m, n, m))  # transition probability

    for x in range(n):
        for y in range(m):
            for i, u in enumerate(U):
                er[i, x, y] += reward((x, y), u, g)
                tp[i, x, y][dynamics((x, y), u, g.shape)] += 1.

    return er, tp


def Q(
    shape: Shape,
    U: List[Action],
    r: np.array,
    p: np.array,
    N: int,
    gamma: float
) -> np.array:
    """
    Q-function mapping.
    """

    n, m = shape

    q = np.zeros((len(U), n, m))

    for _ in tqdm(range(N)):
        expectation = (p * q.max(axis=0)).sum(axis=(3, 4))
        q = r + gamma * expectation

    return q


########
# Main #
########

def main(
    env_pth: str = 'environment.txt',
    gamma: float = 0.99,
    decimals: int = 3,
    output_pth: str = 'output.pdf'
):
    # Load the environment
    env = Environment(env_pth)
    grid = env.to_rewards()

    # Define action space
    U = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Choose N
    eps = 10 ** -decimals
    r_max = grid.max()

    N = math.ceil(math.log((eps / (2 * r_max)) * (1. - gamma) ** 2, gamma))

    # Compute Q_N
    er, tp = mdp(grid, U)

    q = Q(grid.shape, U, er, tp, N, gamma)
    q = q.round(decimals)

    # Compute optimal policy
    mu_star = q.argmax(axis=0)

    # Export result
    env.render(draw=False, what=['pos', 'obj'])

    plt.figure('Environment')
    plt.axis(False)

    n, m = grid.shape
    markers = ['v', '^', '>', '<']
    r_min = grid.min()

    for i, j in itertools.product(range(m), range(n)):
        color = 'red' if grid.T[i, j] == r_min else 'green'
        marker = markers[mu_star.T[i, j]]

        plt.scatter(i, j, c=color, marker=marker, s=25, alpha=0.5)

    plt.savefig(output_pth)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimal policy using reinforcement learning.'
    )

    parser.add_argument(
        '-e',
        '--environment',
        type=str,
        default='environment.txt',
        help='path to environment file'
    )

    parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        default=0.99,
        help='gamma factor'
    )

    parser.add_argument(
        '-d',
        '--decimals',
        type=int,
        default=3,
        help='number of decimals for precision'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output.pdf',
        help='path to output file'
    )

    args = parser.parse_args()

    main(
        env_pth=args.environment,
        gamma=args.gamma,
        decimals=args.decimals,
        output_pth=args.output
    )
