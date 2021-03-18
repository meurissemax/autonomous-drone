#!/usr/bin/env python3

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
import time

from PIL import Image
from tqdm import tqdm
from typing import List

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from analysis.vanishing_point import (  # noqa: E402
    VPClassic,
    VPDetector,
    VPEdgelets
)

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
    results, times = {}, {}

    for method in methods:
        results[method.__name__] = 0
        times[method.__name__] = 0

    # Load evaluation data
    data = []

    with open(json_pth, 'r') as json_file:
        data = json.load(json_file)

    # Evaluate methods on each image
    for d in tqdm(data):
        img = Image.open(d['image'])
        img = np.array(img)

        truth = d['vp']

        for method in methods:
            start = time.time()
            guess = method().detect(img)
            end = time.time()

            dist = math.dist(guess, truth)
            result = 1 if dist < threshold else 0

            results[method.__name__] += result
            times[method.__name__] += end - start

    # Display and export results
    print(results, times)

    os.makedirs(os.path.dirname(output_pth), exist_ok=True)

    for key, value in {'Score': results, 'Time': times}.items():
        x, y = list(value.keys()), list(value.values())

        plt.bar(x, y)

        plt.xlabel('Method')
        plt.ylabel(key)

        plt.grid(False)

        if key == 'Score':
            plt.yticks(range(0, len(data) + 1, int(len(data) / 5)))

        plt.tick_params(axis='x', labelsize=10)

        plt.savefig(os.path.join(output_pth, f'{key.lower()}.pdf'))
        plt.close()


########
# Main #
########

def main(
    json_pth: str = 'data.json',
    threshold: int = 30,
    outpt_pth: str = 'outputs/'
):
    # List of methods
    methods = [VPClassic, VPEdgelets]

    # Evaluate
    evaluate(methods, json_pth, threshold, outpt_pth)


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
        default=30,
        help='threshold for method evaluation'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='outputs/',
        help='path to output file'
    )

    args = parser.parse_args()

    main(
        json_pth=args.input,
        threshold=args.threshold,
        outpt_pth=args.output
    )
