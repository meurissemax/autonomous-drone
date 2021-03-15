#!/usr/bin/env python

"""
Implementation of tools used to generate data sets (a serie of annotated
images) for vanishing point detection methods.
"""

###########
# Imports #
###########

import cv2
import glob
import json
import os

from typing import Dict, List


##########
# Typing #
##########

Dataset = List[Dict]


#####################
# General variables #
#####################

cx, cy = None, None


#############
# Functions #
#############

def _on_click(event, x, y, flags, param):
    """
    Callback function used when a click is performed on the displayed image.

    It saves the position of the click and draw a red circle on it.
    """

    global cx, cy

    if event == cv2.EVENT_LBUTTONDOWN:  # left click
        cx, cy = x, y

        draw = param.copy()

        cv2.circle(draw, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow('Image', draw)


def annotate(inpt_pth: str) -> Dataset:
    """
    Annotate a serie of images with the position of their vanishing points.
    """

    global cx, cy

    # Initialize
    annotations = []

    # Get path of all images
    inpt_pth = os.path.dirname(inpt_pth)
    imgs_pth = glob.glob(f'{inpt_pth}/*.png')

    # Create the image window
    cv2.namedWindow('Image')

    # Annotate each image
    for img_pth in imgs_pth:
        img = cv2.imread(img_pth)

        cv2.setMouseCallback('Image', _on_click, img)

        while True:
            cv2.imshow('Image', img)

            key = cv2.waitKey(0) & 0xFF

            if key == 32:  # space bar
                if cx is not None and cy is not None:
                    annotations.append({
                        'image': img_pth,
                        'vp': [cx, cy]
                    })

                    cx, cy = None, None

                    break

    return annotations


########
# Main #
########

def main(
    inpt_pth: str = 'inputs/',
    json_pth: str = 'data.json'
):
    # Get annotations
    annotations = annotate(inpt_pth)

    # Export
    with open(json_pth, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Data set generation tools for analysis methods.'
    )

    parser.add_argument(
        '-i',
        '--inputs',
        type=str,
        default='inputs/',
        help='path to input images of the data set'
    )

    parser.add_argument(
        '-j',
        '--json',
        type=str,
        default='data.json',
        help='path to export JSON file'
    )

    args = parser.parse_args()

    main(
        inpt_pth=args.inputs,
        json_pth=args.json
    )
