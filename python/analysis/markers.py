"""
Implementation of marker (ArUco, QR code) detection and decoding methods.
"""

###########
# Imports #
###########

import cv2
import math
import numpy as np
import pyzbar.pyzbar as zbar

from abc import ABC, abstractmethod
from cv2 import aruco
from typing import List, Tuple, Union


##########
# Typing #
##########

Image = np.array
Point = List[Union[float, int]]


###########
# Classes #
###########

class MarkerDecoder(ABC):
    """
    Abstract class used to define a marker (detector and) decoder.

    Each marker decoder must inherit this class.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, img: Image) -> Tuple[str, List[Point]]:
        """
        Detect and decode the marker of an image. If there is no marker in the
        image, it returns ('None', 'None'), else it returns content and
        coordinates of the corners.

        Convention for corner coordinates: [bl, br, tr, tl].
        """

        pass

    def ratio(self, img: Image, pts: List[Point]) -> float:
        """
        Compute the ratio between the marker's diagonal length and the image's
        diagonal length.
        """

        # If no marker has been detected
        if pts is None:
            return 0

        # Image dimensions
        size = img.shape

        # Get diagonal lengths
        lengths = [
            math.dist([0, 0], size[:2]),
            math.dist(pts[0], pts[2])
        ]

        # Compute ratio
        ratio = lengths[1] / lengths[0]

        return ratio


# ArUco

class ArucoOpenCV(MarkerDecoder):
    """
    Implementation of an ArUco decoder based on OpenCV ArUco library.
    """

    def __init__(self):
        super().__init__()

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # Abstract interface

    def decode(self, img):
        corners, ids, _ = aruco.detectMarkers(img, self.aruco_dict)

        decoded = None if ids is None else ids[0][0]
        pts = None if ids is None else [list(p) for p in corners[0][0]]

        return decoded, pts


# QR code

class QROpenCV(MarkerDecoder):
    """
    Implementation of a QR code decoder based on OpenCV QR code library.
    """

    def __init__(self):
        super().__init__()

        self.decoder = cv2.QRCodeDetector()

    # Abstract interface

    def decode(self, img):
        methods = [
            self.decoder.detectAndDecode,
            self.decoder.detectAndDecodeCurved
        ]

        for method in methods:
            decoded, pts, straight = method(img)

            if pts is not None:
                break

        decoded = None if pts is None else decoded
        pts = None if pts is None else [list(p) for p in pts[0]]

        return decoded, pts


class QRZBar(MarkerDecoder):
    """
    Implementation of a QR code decoder based on ZBar code library.
    """

    def __init__(self):
        super().__init__()

    # Abstract interface

    def decode(self, img):
        data = zbar.decode(img, symbols=[zbar.ZBarSymbol.QRCODE])

        if len(data) == 0:
            decoded, pts = None, None
        else:
            decoded = data[0].data.decode('utf-8')
            polygon = data[0].polygon

            pts = []

            for p in polygon:
                pts.append([p.x, p.y])

            pts[1], pts[3] = pts[3], pts[1]

        return decoded, pts
