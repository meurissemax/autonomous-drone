#!/usr/bin/env python

"""
Implementation of some tests on the environment representation.
"""

###########
# Imports #
###########

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from uav.environment import Environment  # noqa: E402


########
# Main #
########

def main(
    env_pth: str = 'environment.txt',
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

    # Extract key points
    keypoints = env.extract_keypoints(path, sequence)

    print('Key points')
    print(keypoints)

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
        default='environment.txt',
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
