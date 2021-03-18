#!/usr/bin/env python3

"""
Implementation of the evaluation process of the different QR code decoding
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

from analysis.qr_code import QRDecoder, QROpenCV, QRZBar  # noqa: E402
from plots.latex import plt  # noqa: E402


#############
# Functions #
#############

def evaluate(
    methods: List[QRDecoder],
    json_pth: str,
    output_pth: str
):
    """
    Evaluate a serie of QR code decoders by running them on some truth images.
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

        content = d['content']

        for method in methods:
            start = time.time()
            decoded, _ = method().decode(img)
            end = time.time()

            result = 1 if decoded == content else 0

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
    outpt_pth: str = 'outputs/'
):
    # List of methods
    methods = [QROpenCV, QRZBar]

    # Evaluate
    evaluate(methods, json_pth, outpt_pth)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate QR code decoders.'
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
        json_pth=args.input,
        outpt_pth=args.output
    )
