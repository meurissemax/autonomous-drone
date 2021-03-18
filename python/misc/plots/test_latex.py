#!/usr/bin/env python3

"""
Implementation of some tests with settings used to generate plots with LaTeX
style.
"""

###########
# Imports #
###########

import numpy as np
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from plots.latex import plt  # noqa: E402


#####################
# General variables #
#####################

FUNCTIONS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'exp': np.exp,
    'log': np.log
}


########
# Main #
########

def main(
    start: int = 1,
    stop: int = 100,
    step: float = 0.1,
    function_id: str = 'sin'
):
    # Function
    f = FUNCTIONS.get(function_id, 'sin')

    # Values
    x = np.arange(start, stop, step)
    y = f(x)

    # Figure
    plt.figure('Test')

    plt.plot(x, y)

    plt.xlabel('x')
    plt.ylabel(r'$f(x)$')

    plt.legend([function_id])

    plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate a test plot with LaTeX plot settings.'
    )

    parser.add_argument(
        '-start',
        type=int,
        default=1,
        help='minimum value of x'
    )

    parser.add_argument(
        '-stop',
        type=int,
        default=100,
        help='maximum value of x'
    )

    parser.add_argument(
        '-step',
        type=float,
        default=0.1,
        help='step between each consecutive x value'
    )

    parser.add_argument(
        '-function',
        type=str,
        default='sin',
        choices=list(FUNCTIONS.keys()),
        help='function to plot'
    )

    args = parser.parse_args()

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        function_id=args.function
    )
