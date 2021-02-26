#!/usr/bin/env python

"""
Implementation of functions to generate graphs showing
some data.
"""

###########
# Imports #
###########

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd

from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from typing import List


############
# Settings #
############

plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['figure.autolayout'] = True
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True

if mpl.checkdep_usetex(True):
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True


#####################
# General variables #
#####################

CSV_COL_LIST = ['epoch', 'train_loss_mean', 'train_loss_std']
CSV_COL_PLOT = ['train_loss_mean']
LEGEND = ['Mean']


#############
# Functions #
#############

def line_graph(
    x: list,
    y: List[list],
    x_label: str = 'Epoch',
    y_label: str = 'Loss',
    legend: list = [],
    file_pth: str = 'graph.pdf'
):
    # Initialize figure
    plt.figure()

    try:
        # Create figure
        fig, ax = plt.subplots()

        # Lines
        for y_i in y:
            ax.plot(x, y_i)

        # Axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Legend
        ax.legend(legend)

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#dddddd')

        # Tick parameters
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save
        fig.savefig(file_pth)
    finally:
        plt.close()


########
# Main #
########

def main(
    csv_pth: str = 'data.csv',
    x_label: str = 'Epoch',
    y_label: str = 'Loss',
    file_pth: str = 'graph.pdf'
):
    # Read CSV
    df = pd.read_csv(csv_pth, usecols=CSV_COL_LIST)

    # Graph
    y = [df[col].tolist() for col in CSV_COL_PLOT]
    x = list(range(1, len(y[0]) + 1))

    line_graph(x, y, x_label, y_label, LEGEND, file_pth)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate graph from a CSV file.'
    )

    parser.add_argument(
        '-c',
        '--csv',
        type=str,
        default='data.csv',
        help='path to CSV file'
    )

    parser.add_argument(
        '-x',
        '--xlabel',
        type=str,
        default='Epoch',
        help='label of the X-axis'
    )

    parser.add_argument(
        '-y',
        '--ylabel',
        type=str,
        default='Loss',
        help='label of the Y-axis'
    )

    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default='graph.pdf',
        help='path to output file'
    )

    args = parser.parse_args()

    main(
        csv_pth=args.csv,
        x_label=args.xlabel,
        y_label=args.ylabel,
        file_pth=args.file,
    )
