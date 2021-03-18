#!/usr/bin/env python3

"""
Implementation of the exploration process of the drone.

This process allows the drone to be controlled manually.
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
    TelloEDU,
    Controller
)


#############
# Functions #
#############

def explore(controller: Controller):
    # Initialize the drone
    controller.arm()
    controller.takeoff()

    # Switch to manual control
    controller.manual()

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

    # Exploration
    explore(controller)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Exploration process of the drone.'
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
