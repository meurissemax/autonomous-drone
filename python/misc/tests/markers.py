#!/usr/bin/env python3

"""
Implementation of some tests on the marker decoding methods.
"""

###########
# Imports #
###########

import cv2
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from analysis.markers import ArucoOpenCV, QROpenCV, QRZBar  # noqa: E402


########
# Main #
########

def main(
    method_id: str = 'aruco_opencv',
    input_pth: str = 'marker.png'
):
    # Load method
    methods = {
        'aruco_opencv': ArucoOpenCV,
        'qr_opencv': QROpenCV,
        'qr_zbar': QRZBar
    }

    method = methods.get(method_id)()

    # Open image
    img = cv2.imread(input_pth)

    # Decode marker, if any
    decoded, pts = method.decode(img)

    # Get ratio
    ratio = method.ratio(img, pts)

    # Display result
    if decoded is None:
        print('No marker detected.')
    else:
        print('Marker detected!')
        print(f'Content: {decoded}')
        print(f'Corners: {pts}')
        print(f'Ratio: {ratio}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test marker detecting and decoding methods.'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='aruco_opencv',
        choices=['aruco_opencv', 'qr_opencv', 'qr_zbar'],
        help='method to use'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='marker.png',
        help='path to input file'
    )

    args = parser.parse_args()

    main(
        method_id=args.method,
        input_pth=args.input
    )
