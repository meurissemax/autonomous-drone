#!/usr/bin/env python

"""
Implementation of vanishing point detection methods.
"""

###########
# Imports #
###########

import cv2
import numpy as np
import os

from abc import ABC, abstractmethod
from itertools import product
from PIL import Image as PImage
from typing import List, Tuple, Union


##########
# Typing #
##########

Image = np.array
Point = Tuple
Line = Tuple[Point, Point]
Intermediate = List[Image]


###########
# Classes #
###########

class VPDetector(ABC):
    """
    Abstract class used to define a vanishing point detector.

    Each vanishing point detector must inherit this class.
    """

    def __init__(self, export: bool = False):
        """
        When the 'export' flag is True, the method returns the vanishing point
        and all intermediate results.

        When the 'export' flag is False, the method only returns the vanishing
        point.
        """

        super().__init__()

        self.export = export

    @abstractmethod
    def detect(img: Image) -> Union[Point, Tuple[Point, Intermediate]]:
        """
        Return the coordinates, in pixels, of the vanishing point of the image
        (and eventually all intermediate results).
        """

        pass


class VPClassic(VPDetector):
    """
    Implementation of a vanishing point detector based on classic methods
    (Canny's algorithm and Hough transform).
    """

    def __init__(self, export=False):
        super().__init__(export)

    # Specific

    def _preprocess(self, img: Image) -> Image:
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

    def _edges(self, img: Image) -> Image:
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

    def _lines(self, img: Image) -> List[Line]:
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

    def _intersections(self, lines: List[Line]) -> List[Point]:
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

    # Abstract interface

    def detect(self, img):
        """
        This implementation is largely inspired from:
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

        # Process
        preprocessed = self._preprocess(img)
        edges = self._edges(preprocessed)
        lines = self._lines(edges)
        intersections = self._intersections(lines)

        # Output
        if self.export:
            vp = img.copy()

        # Find best cell
        for i, j in product(range(cols), range(rows)):
            left = i * grid_size
            right = (i + 1) * grid_size
            bottom = j * grid_size
            top = (j + 1) * grid_size

            if self.export:
                cv2.rectangle(vp, (left, bottom), (right, top), (0, 0, 255), 2)

            n_inter = 0

            for x, y in intersections:
                if left < x < right and bottom < y < top:
                    n_inter += 1

            if n_inter > max_inter:
                max_inter = n_inter
                guess = ((left + right) / 2, (bottom + top) / 2)

        if self.export:
            # Draw lines
            img_lines = img.copy()

            for p1, p2 in lines:
                cv2.line(img_lines, p1, p2, (0, 0, 255), 2)

            # Draw best cell
            gx, gy = guess
            mgs = grid_size / 2

            rx1 = int(gx - mgs)
            ry1 = int(gy - mgs)

            rx2 = int(gx + mgs)
            ry2 = int(gy + mgs)

            cv2.rectangle(vp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)

            return guess, [preprocessed, edges, img_lines, vp]

        return guess


########
# Main #
########

def main(
    method_id: str = 'classic',
    img_pth: str = 'image.png',
    export_pth: str = 'results/'
):
    # Load method
    methods = {
        'classic': VPClassic
    }

    method = methods.get(method_id)(export=True)

    # Open image
    img = PImage.open(img_pth)
    img = np.array(img)

    print(f'Image dimensions: {img.shape}')

    # Get results
    guess, exports = method.detect(img)

    # Export results
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

    parser = argparse.ArgumentParser(
        description='Vanishing point detection in an image.'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='classic',
        choices=['classic'],
        help='method to use'
    )

    parser.add_argument(
        '-i',
        '--image',
        type=str,
        default='image.png',
        help='path to the image file'
    )

    parser.add_argument(
        '-e',
        '--export',
        type=str,
        default='results/',
        help='path to the folder for exported results'
    )

    args = parser.parse_args()

    main(
        method_id=args.method,
        img_pth=args.image,
        export_pth=args.export
    )
