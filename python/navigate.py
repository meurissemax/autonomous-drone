#!/usr/bin/env python3

"""
Navigation process of the drone.
"""

###########
# Imports #
###########

from uav.controllers import AirSimDrone, AirSimDroneNoisy, TelloEDU
from uav.environment import Environment
from uav.navigation import (
    NaiveAlgorithm,
    VanishingAlgorithm,
    VisionAlgorithm,
    MarkerAlgorithm,
    DepthAlgorithm
)


########
# Main #
########

def main(
    env_pth: str = 'environment.txt',
    controller_id: str = 'airsim',
    algorithm_id: str = 'naive',
    env_show: bool = False
):
    # Environment
    env = Environment(env_pth)

    # Controller
    controllers = {
        'airsim': AirSimDrone,
        'noisy': AirSimDroneNoisy,
        'telloedu': TelloEDU
    }

    controller = controllers.get(controller_id)()

    # Algorithm
    algorithms = {
        'naive': NaiveAlgorithm,
        'vanishing': VanishingAlgorithm,
        'vision': VisionAlgorithm,
        'marker': MarkerAlgorithm,
        'depth': DepthAlgorithm
    }

    algorithm = algorithms.get(algorithm_id)
    algorithm = algorithm(env, controller, env_show)

    # Navigation
    algorithm.navigate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Navigation process of the drone.'
    )

    parser.add_argument(
        '-e',
        '--environment',
        type=str,
        default='environment.txt',
        help='path to environment file'
    )

    parser.add_argument(
        '-c',
        '--controller',
        type=str,
        default='airsim',
        choices=['airsim', 'noisy', 'telloedu'],
        help='choice of the controller to use'
    )

    parser.add_argument(
        '-a',
        '--algorithm',
        type=str,
        default='naive',
        choices=['naive', 'vanishing', 'vision', 'marker', 'depth'],
        help='navigation algorithm to use'
    )

    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        default=False,
        help='show the environment representation'
    )

    args = parser.parse_args()

    main(
        env_pth=args.environment,
        controller_id=args.controller,
        algorithm_id=args.algorithm,
        env_show=args.show,
    )
