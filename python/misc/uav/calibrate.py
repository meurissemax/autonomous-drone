#!/usr/bin/env python3

"""
Implementation of a simple calibration procedure used to get the focal length
of a camera.

Inspired from:
    - https://bit.ly/3ujRzCL
"""

###########
# Imports #
###########

import numpy as np
import os
import sys

from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from analysis.markers import ArucoOpenCV  # noqa: E402


########
# Main #
########

def main(img_pth: str, distance: float, width: float):
    # Open image
    img = Image.open(img_pth)
    img = np.array(img)

    # Decode marker
    decoded, pts = ArucoOpenCV().decode(img)

    # Get perceived width
    p = pts[2][0] - pts[3][0]

    # Compute focal length
    f = (p * distance) / width

    print(f'Focal length: {f}')

    # Compute distance (to check)
    d = (width * f) / p

    print(f'Distance (in cm): {d}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Simple calibration procedure.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='image.png',
        help='input image with ArUco marker'
    )

    parser.add_argument(
        '-d',
        '--distance',
        type=float,
        help='known distance to marker'
    )

    parser.add_argument(
        '-w',
        '--width',
        type=float,
        help='known width of the marker'
    )

    args = parser.parse_args()

    main(
        img_pth=args.input,
        distance=args.distance,
        width=args.width
    )
