#!/usr/bin/env python

"""
Implementation of some tests on the QR code decoding methods.
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

from analysis.qr_code import QROpenCV  # noqa: E402


########
# Main #
########

def main(
    method_id: str = 'opencv',
    input_pth: str = 'qr.png'
):
    # Load method
    methods = {
        'opencv': QROpenCV
    }

    method = methods.get(method_id)()

    # Open image
    img = cv2.imread(input_pth)

    # Decode QR code, if any
    decoded, pts = method.decode(img)

    # Display result
    if decoded is None:
        print('No QR code detected.')
    else:
        print('QR code detected!')
        print(f'Content: {decoded}')
        print(f'Corners: {pts}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test QR code detecting and decoding methods.'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='opencv',
        choices=['opencv'],
        help='method to use'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='qr.png',
        help='path to input file'
    )

    args = parser.parse_args()

    main(
        method_id=args.method,
        input_pth=args.input
    )
