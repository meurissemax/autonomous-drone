#!/usr/bin/env python3

"""
Implementation of the main procedure of supervised learning. It trains and
evaluates a model.
"""

###########
# Imports #
###########

import os

from datetime import datetime

from learning.evaluation import evaluate
from learning.training import train


########
# Main #
########

def main(
    outputs_pth: str = 'outputs/',
    criterion_id: str = 'mse',
    dataset_id: str = 'class',
    train_pth: str = 'train.json',
    model_id: str = 'densenet121',
    augment: bool = False,
    edges: bool = False,
    batch_size: int = 32,
    out_channels: int = 2,
    num_epochs: int = 30,
    test_pth: str = 'test.json',
    metric_id: str = 'pr'
):
    # Create output folder
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    folder_pth = os.path.join(outputs_pth, now)
    folder_pth += '/'

    # Get weights path
    weights_pth = os.path.join(folder_pth, f'{model_id}.pth')

    # Train
    train(
        outputs_pth=folder_pth,
        criterion_id=criterion_id,
        dataset_id=dataset_id,
        train_pth=train_pth,
        model_id=model_id,
        augment=augment,
        edges=edges,
        batch_size=batch_size,
        out_channels=out_channels,
        num_epochs=num_epochs,
        weights_pth=weights_pth
    )

    # Evaluate
    evaluate(
        outputs_pth=folder_pth,
        dataset_id=dataset_id,
        test_pth=test_pth,
        model_id=model_id,
        edges=edges,
        batch_size=batch_size,
        out_channels=out_channels,
        weights_pth=weights_pth,
        metric_id=metric_id
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train and evaluate a deep learning model.'
    )

    parser.add_argument(
        '-outputs',
        type=str,
        default='outputs/',
        help='path to outputs folder'
    )

    parser.add_argument(
        '-criterion',
        type=str,
        default='mse',
        choices=['mse', 'nll'],
        help='criterion to use'
    )

    parser.add_argument(
        '-dataset',
        type=str,
        default='class',
        choices=['class', 'image'],
        help='data set to use'
    )

    parser.add_argument(
        '-train',
        type=str,
        default='train.json',
        help='path to JSON file with training data'
    )

    parser.add_argument(
        '-model',
        type=str,
        default='densenet121',
        choices=['densenet121', 'densenet161', 'small', 'unet'],
        help='model to train'
    )

    parser.add_argument(
        '-augment',
        default=False,
        action='store_true',
        help='flag to enable data augmentation'
    )

    parser.add_argument(
        '-edges',
        default=False,
        action='store_true',
        help='flag to work with edges'
    )

    parser.add_argument(
        '-batch',
        type=int,
        default=32,
        help='batch size'
    )

    parser.add_argument(
        '-channels',
        type=int,
        default=2,
        help='number output channels'
    )

    parser.add_argument(
        '-epochs',
        type=int,
        default=30,
        help='number of epochs'
    )

    parser.add_argument(
        '-test',
        type=str,
        default='test.json',
        help='path to JSON file with testing data'
    )

    parser.add_argument(
        '-metric',
        type=str,
        default='pr',
        choices=['pr'],
        help='metric to use'
    )

    args = parser.parse_args()

    main(
        outputs_pth=args.outputs,
        criterion_id=args.criterion,
        dataset_id=args.dataset,
        train_pth=args.train,
        model_id=args.model,
        augment=args.augment,
        edges=args.edges,
        batch_size=args.batch,
        out_channels=args.channels,
        num_epochs=args.epochs,
        test_pth=args.test,
        metric_id=args.metric
    )
