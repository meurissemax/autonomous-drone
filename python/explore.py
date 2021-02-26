#!/usr/bin/env python

"""
Implementation of the exploration process of the drone.
"""

###########
# Imports #
###########

from controllers import (
    AirSimDrone,
    AirSimDroneNoisy,
    Controller,
    TelloEDU
)

from environment import Environment


#############
# Functions #
#############

def explore(
    env: Environment,
    controller: Controller,
    show: bool = False
):
    # Determine shortest path to objective
    path = env.path()

    # Show the path
    if show:
        env.render(path=path, what=['pos', 'obj'])

    # Initialize the drone
    controller.arm()
    controller.takeoff()

    # Switch to manual control
    controller.manual()

    # When drone exits manual control, stop it
    controller.land()
    controller.disarm()

    # Keep the environment
    if show:
        env.keep()


########
# Main #
########

def main(
    env_pth: str = 'indoor-corridor.txt',
    controller_id: str = 'airsim',
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

    # Exploration
    explore(env, controller, env_show)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Exploration process of the drone.')

    parser.add_argument('-e', '--environment', type=str, default='indoor-corridor.txt', help='path to environment file')
    parser.add_argument('-c', '--controller', type=str, default='airsim', choices=['airsim', 'noisy', 'telloedu'], help='choice of the controller to use')
    parser.add_argument('-s', '--show', action='store_true', default=False, help='show the environment representation')

    args = parser.parse_args()

    main(
        env_pth=args.environment,
        controller_id=args.controller,
        env_show=args.show
    )
