#!/usr/bin/env python

"""
Implementation of some test on the vanishing point detection methods.
"""

###########
# Imports #
###########

import cv2
import numpy as np
import os
import sys

from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from analysis.vanishing_point import VPClassic, VPEdgelets  # noqa: E402


########
# Main #
########

def main(
    method_id: str = 'classic',
    inpt_pth: str = 'image.png',
    outpt_pth: str = 'results/'
):
    # Load method
    methods = {
        'classic': VPClassic,
        'edgelets': VPEdgelets
    }

    method = methods.get(method_id)(export=True)

    # Open image
    img = Image.open(inpt_pth)
    img = np.array(img)

    print(f'Image dimensions: {img.shape}')

    # Get results
    guess, exports = method.detect(img)

    # Export results
    os.makedirs(os.path.dirname(outpt_pth), exist_ok=True)

    for i, export in enumerate(exports):
        cv2.imwrite(
            os.path.join(outpt_pth, f'{i}.png'),
            export
        )

    # Display vanishing point guess
    print(f'Vanishing point guess: {guess}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test or evaluate vanishing point detectors.'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='classic',
        choices=['classic', 'edgelets'],
        help='method to use'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='image.png',
        help='path to input file'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='results/',
        help='path to output file or folder'
    )

    args = parser.parse_args()

    main(
        method_id=args.method,
        inpt_pth=args.input,
        outpt_pth=args.output
    )
