#!/usr/bin/env python

"""
Implementation of the evaluation procedure of the deep
learning models.
"""

###########
# Imports #
###########

import csv
import numpy as np
import os
import torch

from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable

from dataset import IndoorDataset
from models import DenseNet161


##########
# Typing #
##########

Tensors = Iterable[torch.Tensor]


#############
# Functions #
#############

def pr_eval(outputs: Tensors, targets: Tensors) -> list:
    adapt = lambda t: torch.flatten(torch.argmax(t, dim=1))

    outputs = adapt(outputs)
    targets = adapt(targets)

    args = {
        'y_true': targets,
        'y_pred': outputs,
        'average': 'weighted',
        'zero_division': 0
    }

    p = precision_score(**args)
    r = recall_score(**args)

    return [p, r]


########
# Main #
########

def main(
    outputs_pth: str = 'outputs/',
    test_pth: str = 'test.json',
    batch_size: int = 32,
    num_workers: int = 0,
    model_id: str = 'densenet161',
    weights_pth: str = 'weights.pth',
    metric_id: str = 'pr'
):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f'Device: {device}')

    # Output folder
    os.makedirs(os.path.dirname(outputs_pth), exist_ok=True)

    # Data set and data loader
    print('Loading data set...')

    testset = IndoorDataset(test_pth, model_id)
    loader = DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device == 'cuda'
    )

    # Model
    models = {
        'densenet161': DenseNet161
    }

    inpt, trgt = testset[0]

    in_channels = inpt.size()[0]
    out_channels = trgt.size()[0]

    model = models.get(model_id)(in_channels, out_channels)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_pth, map_location=device))
    model.eval()

    # Evaluation
    metrics = {
        'pr': pr_eval
    }

    evaluate = metrics.get(metric_id)

    values = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            metric = evaluate(outputs.cpu(), targets.cpu())
            values.append(metric)

    # Save data
    stats_pth = os.path.join(outputs_pth, 'evaluate.csv')

    stats_headers = {
        'pr': ['precision_mean', 'recall_mean', 'precision_std', 'recall_std']
    }

    stats_header = stats_headers.get(metric_id)

    metric_mean = np.mean(values, axis=0)
    metric_std = np.std(values, axis=0)

    with open(stats_pth, 'w', newline='') as f:
        csv.writer(f).writerow(stats_header)
        csv.writer(f).writerow(list(np.concatenate((metric_mean, metric_std))))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a deep learning model.')

    parser.add_argument('-o', '--outputs', type=str, default='outputs/', help='path to outputs folder')
    parser.add_argument('-t', '--test', type=str, default='test.json', help='path to JSON file with testing data')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-w', '--workers', type=int, default=0, help='number of workers')
    parser.add_argument('-m', '--model', type=str, default='densenet161', choices=['densenet161'], help='model to evaluate')
    parser.add_argument('-e', '--weights', type=str, default='weights.pth', help='path to weights file')
    parser.add_argument('-r', '--metric', type=str, default='pr', choices=['pr'], help='metric to use')

    args = parser.parse_args()

    main(
        outputs_pth=args.outputs,
        test_pth=args.test,
        batch_size=args.batch,
        num_workers=args.workers,
        model_id=args.model,
        weights_pth=args.weights,
        metric_id=args.metric
    )
