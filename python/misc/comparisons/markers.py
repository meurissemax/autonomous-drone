#!/usr/bin/env python3

"""
Implementation of the evaluation process of the different marker decoding
methods.
"""

###########
# Imports #
###########

import json
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

from analysis.markers import (  # noqa: E402
    MarkerDecoder,
    ArucoOpenCV,
    QROpenCV,
    QRZBar
)

from plots.latex import plt  # noqa: E402


#############
# Functions #
#############

def evaluate(
    methods: List[MarkerDecoder],
    json_pth: str,
    output_pth: str
):
    """
    Evaluate a serie of marker decoders by running them on some truth images.
    """

    # Initialize
    results, times = {}, {}

    for method in methods:
        results[method.__name__] = 0
        times[method.__name__] = []

    # Load evaluation data
    data = []

    with open(json_pth, 'r') as json_file:
        data = json.load(json_file)

    # Evaluate methods on each image
    for d in tqdm(data):
        img = Image.open(d['image'])
        img = np.array(img)

        content = d['content']

        for method in methods:
            start = time.time()
            decoded, _ = method().decode(img)
            end = time.time()

            result = 1 if str(decoded) == content else 0

            results[method.__name__] += result
            times[method.__name__].append(end - start)

    for key, value in times.items():
        times[key] = np.mean(value)

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
    marker_id: str = 'aruco',
    json_pth: str = 'data.json',
    outpt_pth: str = 'outputs/'
):
    # List of methods
    lists = {
        'aruco': [ArucoOpenCV],
        'qr': [QROpenCV, QRZBar]
    }

    methods = lists.get(marker_id)

    # Evaluate
    evaluate(methods, json_pth, outpt_pth)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate marker decoders.'
    )

    parser.add_argument(
        '-m',
        '--marker',
        type=str,
        default='aruco',
        choices=['aruco', 'qr'],
        help='Marker decoders to evaluate'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='data.json',
        help='path to JSON data file'
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
        marker_id=args.marker,
        json_pth=args.input,
        outpt_pth=args.output
    )
