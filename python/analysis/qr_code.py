#!/usr/bin/env python

"""
Implementation of QR code detection and decoding methods.
"""

###########
# Imports #
###########

import cv2
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple


##########
# Typing #
##########

Image = np.array
Point = List[float]


###########
# Classes #
###########

class QRDecoder(ABC):
    """
    Abstract class used to define a QR code (detector and) decoder.

    Each QR code decoder must inherit this class.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, img: Image) -> Tuple[str, List[Point]]:
        """
        Detect and decode the QR code of an image. If there is no QR code in
        the image, it returns ('None', 'None'), else it returns content and
        coordinates of the corners.
        """

        pass


class QROpenCV(QRDecoder):
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
        pts = None if pts is None else pts[0]

        return decoded, pts


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
