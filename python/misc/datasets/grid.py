#!/usr/bin/env python3

"""
Implementation of a tool used to convert vanishing point coordinate to the
corresponding cell in the grid splitting the image.
"""

###########
# Imports #
###########

import cv2
import json
import numpy as np

from itertools import product
from tqdm import tqdm
from typing import Dict, List


##########
# Typing #
##########

Data = List[Dict]
Image = np.array


#############
# Functions #
#############

def _target(img: Image, n: int, coord: List) -> List:
    """
    Define in which cell of the grid the point 'coord' is located.

    It returns a vector with binary values indicating, for each cell, if it
    contains the point or not.

    Example : [0, 1, 0, 0] means that the second cell of the grid contains the
    vanishing point (counted row by row, from left to right).
    """

    # Dimensions of the image
    h, w, _ = img.shape

    # Dimensions of each cell
    cell = (w // n, h // n)

    # Iterate over each cell
    x, y = coord[0], coord[1]
    inside = None

    for idx, (j, i) in enumerate(product(range(n), range(n))):
        left = i * cell[0]
        right = (i + 1) * cell[0]
        bottom = j * cell[1]
        top = (j + 1) * cell[1]

        if left < x <= right and bottom < y <= top:
            inside = idx

            break

    # Create binary vector
    binary = [0] * (n ** 2)

    if inside is not None:
        binary[inside] = 1

    return binary


def convert(coordinates: Data, n: int = 5) -> Data:
    # Initialize
    cells = []

    # Determine cell for each image
    for data in tqdm(coordinates):
        img_pth = data['image']
        coord = data['vp']

        img = cv2.imread(img_pth)

        cells.append({
            'image': img_pth,
            'target': _target(img, n, coord)
        })

    return cells


########
# Main #
########

def main(
    inpt_pth: str = 'coordinates.json',
    n: int = 5,
    outpt_pth: str = 'cells.json'
):
    # Get coordinates
    coordinates = []

    with open(inpt_pth, 'r') as json_file:
        coordinates = json.load(json_file)

    # Get corresponding cells
    cells = convert(coordinates, n)

    # Export
    with open(outpt_pth, 'w') as json_file:
        json.dump(cells, json_file, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert vanishing point coordinates to cell.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='coordinates.json',
        help='path to input JSON file with coordinates'
    )

    parser.add_argument(
        '-n',
        type=int,
        default=5,
        help='grid size (n x n)'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='cells.json',
        help='path to output JSON file'
    )

    args = parser.parse_args()

    main(
        inpt_pth=args.input,
        n=args.n,
        outpt_pth=args.output
    )
