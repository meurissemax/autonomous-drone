#!/usr/bin/env python3

"""
Implementation of tools used to generate data sets (a serie of annotated
images) for QR code decoding methods.
"""

###########
# Imports #
###########

import glob
import json

from tqdm import tqdm
from typing import Dict, List


##########
# Typing #
##########

Dataset = List[Dict]


#############
# Functions #
#############

def annotate(imgs_pth: str, content: str) -> Dataset:
    # Get image list
    imgs_list = glob.glob(f'{imgs_pth}*.png', recursive=True)

    # Initialize annotations
    annotations = []

    # Annotate using intervals
    for img in tqdm(imgs_list):
        annotations.append({
            'image': img,
            'content': content
        })

    return annotations


########
# Main #
########

def main(
    imgs_pth: str = 'images/',
    content: str = '',
    json_pth: str = 'data.json'
):
    # Get annotations
    annotations = annotate(imgs_pth, content)

    # Export annotations
    with open(json_pth, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Data set generation tools for QR code decoding methods.'
    )

    parser.add_argument(
        '-i',
        '--inputs',
        type=str,
        default='images/',
        help='path to input images of the data set'
    )

    parser.add_argument(
        '-c',
        '--content',
        type=str,
        default='',
        help='content of the QR codes'
    )

    parser.add_argument(
        '-j',
        '--json',
        type=str,
        default='data.json',
        help='path to JSON data file'
    )

    args = parser.parse_args()

    main(
        imgs_pth=args.inputs,
        content=args.content,
        json_pth=args.json
    )
