"""
Implementation of vanishing point detection methods.
"""

###########
# Imports #
###########

import cv2
import numpy as np

from abc import ABC, abstractmethod
from itertools import product
from skimage import color, feature, transform
from typing import List, Tuple, Union


##########
# Typing #
##########

Image = np.array

Point = Tuple
Line = Tuple[Point, Point]

Edgelets = Tuple[np.array]

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
    def detect(self, img: Image) -> Union[Point, Tuple[Point, Intermediate]]:
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
        median = np.median(img)
        sigma = 0.33

        lo_thresh = int(max(0, (1.0 - sigma) * median))
        hi_thresh = int(min(255, (1.0 + sigma) * median))
        filter_size = 3

        img = cv2.Canny(
            img,
            lo_thresh,
            hi_thresh,
            apertureSize=filter_size,
            L2gradient=True
        )

        return img

    def _lines(self, img: Image) -> List[Line]:
        h, w = img.shape

        # Hough transform
        rho = 1
        theta = np.pi / 180
        thresh = 10
        min_line_length = w // 40
        max_line_gap = w // 256

        lines = cv2.HoughLinesP(
            img,
            rho,
            theta,
            thresh,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # Get and filter end points
        if lines is None:
            return []

        xy_thresh = w // 25
        pts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(x1 - x2) > xy_thresh and abs(y1 - y2) > xy_thresh:
                pts.append(((x1, y1), (x2, y2)))

        return pts

    def _intersections(self, lines: List[Line]) -> List[Point]:
        if len(lines) == 0:
            return []

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
        if len(intersections) > 0:
            for i, j in product(range(cols), range(rows)):
                left = i * grid_size
                right = (i + 1) * grid_size
                bottom = j * grid_size
                top = (j + 1) * grid_size

                if self.export:
                    cv2.rectangle(
                        vp,
                        (left, bottom),
                        (right, top),
                        (0, 0, 255),
                        2
                    )

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

            if len(lines) > 0:
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


class VPEdgelets(VPDetector):
    """
    Implementation of a vanishing point detector based on edgelets and RANSAC.

    Inspired from:
        - Auto-Rectification of User Photos, Chaudhury et al.
        - https://github.com/chsasank/Image-Rectification
    """

    def __init__(self, export=False):
        super().__init__(export)

    # Specific

    def _edgelets(self, img: Image) -> Edgelets:
        """
        Compute edgelets of an image.
        """

        # Get lines
        gray = color.rgb2gray(img)
        edges = feature.canny(gray, 3)
        lines = transform.probabilistic_hough_line(
            edges,
            line_length=3,
            line_gap=2
        )

        locations, directions, strengths = [], [], []

        for p0, p1 in lines:
            p0, p1 = np.array(p0), np.array(p1)

            locations.append((p0 + p1) / 2)
            directions.append(p1 - p0)
            strengths.append(np.linalg.norm(p1 - p0))

        # Normalize
        locations = np.array(locations)
        directions = np.array(directions)
        strengths = np.array(strengths)

        norm = np.linalg.norm(directions, axis=1)[:, np.newaxis]
        directions = np.divide(directions, norm)

        return locations, directions, strengths

    def _lines(self, edgelets: Edgelets) -> np.array:
        """
        Compute lines from edgelets.
        """

        locations, directions, _ = edgelets

        normals = np.zeros_like(directions)
        normals[:, 0] = directions[:, 1]
        normals[:, 1] = -directions[:, 0]

        p = -np.sum(locations * normals, axis=1)
        lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)

        return lines

    def _votes(self, edgelets: Edgelets, model: np.array) -> int:
        """
        Compute votes for each of the edgelet against a given vanishing point.
        """

        threshold_inlier = 5

        vp = model[:2] / model[2]
        locations, directions, strengths = edgelets

        est_directions = locations - vp

        dot_prod = np.sum(est_directions * directions, axis=1)

        abs_prod = np.linalg.norm(directions, axis=1)
        abs_prod *= np.linalg.norm(est_directions, axis=1)
        abs_prod[abs_prod == 0] = 1e-5

        cosine_theta = dot_prod / abs_prod
        theta = np.arccos(np.clip(-1, 1, np.abs(cosine_theta)))

        theta_thresh = threshold_inlier * np.pi / 180

        return (theta < theta_thresh) * strengths

    # Abstract interface

    def detect(self, img):
        """
        Estimate vanishing point using edgelets and RANSAC.
        """

        num_ransac_iter = 50

        # Compute edgelets
        edgelets = self._edgelets(img)
        _, _, strengths = edgelets

        # Compute lines
        lines = self._lines(edgelets)

        num_pts = strengths.size

        arg_sort = np.argsort(-strengths)
        first_index_space = arg_sort[:num_pts // 5]
        second_index_space = arg_sort[:num_pts // 2]

        # Compute best guess
        best_model = np.array([0, 0, 1])
        best_votes = np.zeros(num_pts)

        for _ in range(num_ransac_iter):
            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

            current_model = np.cross(lines[ind1], lines[ind2])

            # Rejecte degenerate cases
            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                continue

            current_votes = self._votes(edgelets, current_model)

            if current_votes.sum() > best_votes.sum():
                best_model = current_model
                best_votes = current_votes

        guess = tuple(best_model[:2] / best_model[2])

        if self.export:
            vp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.circle(vp, (int(guess[0]), int(guess[1])), 5, (0, 255, 0), -1)

            return guess, [vp]

        return guess
