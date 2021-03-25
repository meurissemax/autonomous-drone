#!/usr/bin/env python3

"""
Implementation of a tool used to split a data set into training and testing
sets.
"""

###########
# Imports #
###########

import json
import os
import random

from typing import Dict, List, Tuple


##########
# Typing #
##########

Dataset = List[Dict]


#############
# Functions #
#############

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


########
# Main #
########

def main(
    inpt_pth: str = 'dataset.json',
    ratio: int = 0.7,
    outpt_pth: str = 'outputs/'
):
    # Get data set
    dataset = []

    with open(inpt_pth, 'r') as json_file:
        dataset = json.load(json_file)

    # Split data set
    train, test = split(dataset, ratio)

    # Export sets
    for data, name in zip([train, test], ['train', 'test']):
        pth = os.path.join(outpt_pth, f'{name}.json')

        with open(pth, 'w') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Split data set into training and testing sets.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='dataset.json',
        help='path to data set JSON file'
    )

    parser.add_argument(
        '-r',
        '--ratio',
        type=float,
        default=0.7,
        help='ratio of the training set'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='outputs/',
        help='path to folder to export JSON files'
    )

    args = parser.parse_args()

    main(
        inpt_pth=args.input,
        ratio=args.ratio,
        outpt_pth=args.output
    )
