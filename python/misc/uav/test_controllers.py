#!/usr/bin/env python3

"""
Implementation of some tests with controllers.
"""

###########
# Imports #
###########

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from uav.controllers import (  # noqa: E402
    AirSimDrone,
    AirSimDroneNoisy,
    Controller,
    TelloEDU
)


#############
# Functions #
#############

def trajectory(controller: Controller):
    # Initialize the drone
    controller.arm()
    controller.takeoff()

    # Follow simple trajectory
    for _ in range(4):
        controller.move('forward', 100, 50)
        controller.rotate('cw', 180)

    # When drone exits manual control, stop it
    controller.land()
    controller.disarm()


########
# Main #
########

def main(
    controller_id: str = 'airsim'
):
    # Controller
    controllers = {
        'airsim': AirSimDrone,
        'noisy': AirSimDroneNoisy,
        'telloedu': TelloEDU
    }

    controller = controllers.get(controller_id)()

    # Trajectory
    trajectory(controller)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Simple test trajectory with controllers.'
    )

    parser.add_argument(
        '-c',
        '--controller',
        type=str,
        default='airsim',
        choices=['airsim', 'noisy', 'telloedu'],
        help='choice of the controller to use'
    )

    args = parser.parse_args()

    main(
        controller_id=args.controller
    )
