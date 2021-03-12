#!/usr/bin/env python

"""
Implementation of the evaluation process of the different vanishing point
detection methods.
"""

###########
# Imports #
###########

import json
import math
import numpy as np
import os
import sys

from PIL import Image
from tqdm import tqdm
from typing import List

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from analysis.vanishing_point import VPClassic, VPDetector  # noqa: E402
from plots.latex import plt  # noqa: E402


#############
# Functions #
#############

def evaluate(
    methods: List[VPDetector],
    json_pth: str,
    threshold: int,
    output_pth: str
):
    """
    Evaluate a serie of vanishing point detectors by running them on some truth
    images.
    """

    # Initialize
    results = {}

    for method in methods:
        results[method.__name__] = []

    # Load evaluation data
    data = []

    with open(json_pth, 'r') as json_file:
        data = json.load(json_file)

    # Evaluate methods on each image
    for d in tqdm(data):
        img = Image.open(d['image'])
        img = np.array(img)

        for method in methods:
            guess = method().detect(img)
            truth = d['vp']

            dist = math.dist(guess, truth)
            result = 1 if dist < threshold else 0

            results[method.__name__].append(result)

    # Get the score of each method
    for key, values in results.items():
        results[key] = sum(values)

    # Display and export results
    print(results)

    x, y = list(results.keys()), list(results.values())

    plt.bar(x, y)

    plt.xlabel('Method')
    plt.ylabel('Score')

    plt.grid(False)
    plt.yticks(range(0, len(data) + 1, int(len(data) / 5)))
    plt.tick_params(axis='x', labelsize=10)

    plt.savefig(output_pth)
    plt.close()


########
# Main #
########

def main(
    inpt_pth: str = 'data.json',
    threshold: int = 15,
    outpt_pth: str = 'output.pdf'
):
    # List of methods
    methods = [VPClassic]

    # Evaluate
    evaluate(methods, inpt_pth, threshold, outpt_pth)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate vanishing point detectors.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='data.json',
        help='path to JSON data file'
    )

    parser.add_argument(
        '-t',
        '--threshold',
        type=int,
        default=15,
        help='threshold for method evaluation'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output.pdf',
        help='path to output file'
    )

    args = parser.parse_args()

    main(
        inpt_pth=args.input,
        threshold=args.threshold,
        outpt_pth=args.output
    )
