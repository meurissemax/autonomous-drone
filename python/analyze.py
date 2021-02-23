#!/usr/bin/env python

"""
Implementation of analysis methods based on
environment information.
"""

###########
# Imports #
###########

import cv2
import numpy as np
import os

from itertools import product
from PIL import Image


#############
# Functions #
#############

def _preprocess(img):
    # Bilateral filtering
    d = 10
    sigma_color = 10
    sigma_space = 100

    img = cv2.bilateralFilter(
        img,
        d,
        sigma_color,
        sigma_space
    )

    return img


def _edges(img):
    # Canny's algorithm
    lo_thresh = 50
    hi_thresh = 250
    filter_size = 3

    img = cv2.Canny(
        img,
        lo_thresh,
        hi_thresh,
        apertureSize=filter_size,
        L2gradient=True
    )

    # Gaussian blur
    blur_size = 3

    img = cv2.GaussianBlur(
        img,
        (blur_size, blur_size),
        0
    )

    return img


def _lines(img):
    # Hough transform
    rho = 1
    theta = np.pi / 180
    thresh = 10
    min_line_length = 15
    max_line_gap = 5

    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Get and filter end points
    x_thresh = 35
    y_thresh = 20

    pts = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(x1 - x2) > x_thresh and abs(y1 - y2) > y_thresh:
            pts.append(((x1, y1), (x2, y2)))

    return pts


def _intersections(lines):
    inters = []

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for i, l1 in enumerate(lines):
        for l2 in lines[i + 1:]:
            if not l1 == l2:
                x_diff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
                y_diff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

                div = det(x_diff, y_diff)

                if div == 0:
                    continue

                d = (det(*l1), det(*l2))

                x = det(d, x_diff) / div
                y = det(d, y_diff) / div

                inters.append((x, y))

    return inters


def vanishing_point(img, export=False):
    """
    Return the coordinates, in pixels, of the vanishing point
    of the image.

    This function is largely inspired from:
        - https://github.com/SZanlongo/vanishing-point-detection
    """

    # Color convention
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Image dimensions
    h, w, _ = img.shape

    # Grid dimensions
    grid_size = min(h, w) // 5

    rows = (h // grid_size) + 1
    cols = (w // grid_size) + 1

    # Initialize guess (cell with most intersections)
    max_inter = 0
    guess = (0.0, 0.0)

    # Get intersections
    preprocessed = _preprocess(img)
    edges = _edges(preprocessed)
    lines = _lines(edges)
    intersections = _intersections(lines)

    # Find best cell
    for i, j in product(range(cols), range(rows)):
        left = i * grid_size
        right = (i + 1) * grid_size
        bottom = j * grid_size
        top = (j + 1) * grid_size

        if export:
            cv2.rectangle(img, (left, bottom), (right, top), (0, 0, 255), 2)

        n_inter = 0

        for x, y in intersections:
            if left < x < right and bottom < y < top:
                n_inter += 1

        if n_inter > max_inter:
            max_inter = n_inter
            guess = ((left + right) / 2, (bottom + top) / 2)

    # Draw best cell
    if export:
        gx, gy = guess
        mgs = grid_size / 2

        rx1 = int(gx - mgs)
        ry1 = int(gy - mgs)

        rx2 = int(gx + mgs)
        ry2 = int(gy + mgs)

        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)

        return preprocessed, edges, lines, img, guess

    return guess


########
# Main #
########

def main(
    img_pth='image.png',
    export_pth='analysis/'
):
    # Open image
    img = Image.open(img_pth)
    img = np.array(img)

    print(f'Image dimensions: {img.shape}')

    # Get results
    preprocessed, edges, lines, vp, guess = vanishing_point(img, export=True)

    # Draw lines
    img_lines = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for p1, p2 in lines:
        cv2.line(img_lines, p1, p2, (0, 0, 255), 2)

    # Export results
    exports = [preprocessed, edges, img_lines, vp]

    os.makedirs(os.path.dirname(export_pth), exist_ok=True)

    for i, export in enumerate(exports):
        cv2.imwrite(
            os.path.join(export_pth, f'{i}.png'),
            export
        )

    # Display vanishing point guess
    print(f'Vanishing point guess: {guess}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analysis of an image.')

    parser.add_argument('-i', '--image', type=str, default='image.png', help='path to the image file')
    parser.add_argument('-e', '--export', type=str, default='analysis/', help='path to the folder for exported results')

    args = parser.parse_args()

    main(
        img_pth=args.image,
        export_pth=args.export
    )
