#!/usr/bin/env python

"""
Implementation of tools used to generate a data set
(annotate images and split intro training and testing
sets).
"""

###########
# Imports #
###########

import glob
import json
import os
import random

from tqdm import tqdm
from typing import Dict, List, Tuple


##########
# Typing #
##########

Dataset = List[Dict]


#####################
# General variables #
#####################

"""
Intervals used to annotate images. Each interval is a tuple
composed of
    - target (in format directly useable by PyTorch);
    - ID lower bound;
    - ID upper bound.

Each interval is defined by [lower_bound, upper_bound]. The
following example will annotate each image whose ID is in
[0, 100] by [1, 0] and each image whose ID is in [101, 200]
by [0, 1].

The ID of an image must be at the end of the image name,
separed by an underscore, e.g. img_1.png.
"""

INTERVALS = [
    ([1, 0], 0, 100),
    ([0, 1], 101, 200),
]


#############
# Functions #
#############

def _extract(img_name: str) -> int:
    """
    Extract the ID from the image name.
    """

    img_id = os.path.splitext(img_name)[0]
    img_id = os.path.basename(img_id)
    img_id = img_id.split('_')[-1]
    img_id = int(img_id)

    return img_id


def _list(imgs_pth: str) -> List[Tuple[str, int]]:
    """
    Construct a list of image paths with associated ID
    based on images path.
    """

    # Initialize (image, ID) list
    imgs_list = []

    # Get path of all images
    imgs_pth = os.path.dirname(imgs_pth)
    imgs = glob.glob(f'{imgs_pth}/**/*.png', recursive=True)

    # Iterate over each image
    for img in imgs:
        imgs_list.append((
            img,
            _extract(img)
        ))

    return imgs_list


def _target(img_id: int) -> list:
    """
    Get the target according to the ID.
    """

    for target, lower, upper in INTERVALS:
        if img_id >= lower and img_id <= upper:
            return target

    return target


def annotate(imgs_pth: str) -> Dataset:
    # Get (image, ID) list
    imgs_list = _list(imgs_pth)

    # Initialize annotations
    annotations = []

    # Iterate over each image
    for img, img_id in tqdm(imgs_list):
        annotations.append({
            'image': img,
            'target': _target(img_id)
        })

    return annotations


def split(dataset: Dataset, ratio: int) -> Tuple[Dataset, Dataset]:
    """
    Split a data set into training and testing sets.
    """

    n = len(dataset)

    # Get indexes of training items
    n_train = int(n * max(0, min(ratio, 1)))
    idxs_train = random.sample(range(n), n_train)

    # Get indexes of testing items
    idxs_test = [i for i in range(n) if i not in idxs_train]

    # Create training and testing sets
    train = [dataset[idx] for idx in idxs_train]
    test = [dataset[idx] for idx in idxs_test]

    return train, test


def export(dataset: Dataset, json_pth: str, fname: str):
    """
    Export a data set to a JSON file.
    """

    pth = os.path.join(json_pth, fname)

    with open(pth, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)


########
# Main #
########

def main(
    imgs_pth: str = 'images/',
    ratio: int = 0.7,
    json_pth: str = 'dataset/'
):
    # Get annotations
    annotations = annotate(imgs_pth)

    # Split data set
    train, test = split(annotations, ratio)

    # Export sets
    export(train, json_pth, 'train.json')
    export(test, json_pth, 'test.json')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Data set generation tools.'
    )

    parser.add_argument(
        '-i',
        '--images',
        type=str,
        default='images/',
        help='path to images of the data set'
    )

    parser.add_argument(
        '-r',
        '--ratio',
        type=float,
        default=0.7,
        help='ratio of the training set'
    )

    parser.add_argument(
        '-j',
        '--json',
        type=str,
        default='dataset/',
        help='path to export JSON files'
    )

    args = parser.parse_args()

    main(
        imgs_pth=args.images,
        ratio=args.ratio,
        json_pth=args.json
    )
