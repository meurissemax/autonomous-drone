"""
Implementation of the evaluation procedure of the deep learning models.
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

from .datasets import ClassDataset, ImageDataset
from .models import DenseNet161, SmallConvNet, UNet


##########
# Typing #
##########

Tensors = Iterable[torch.Tensor]


#############
# Functions #
#############

# Utilities

def _to_class(t: Tensors) -> Tensors:
    return torch.argmax(t, dim=1)


def _tensorify(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor(t.size())


def _flatten(t: Tensors) -> Tensors:
    n = len(list(t.size()))

    return torch.flatten(torch.argmax(t, dim=1)) if n > 1 else t


# Evaluation

def pr_eval(outputs: Tensors, targets: Tensors) -> list:
    """
    Compute precision and recall evaluation metrics.
    """

    # Transform probabilities to class
    transform = torch.equal(
        _tensorify(outputs),
        _tensorify(targets)
    )

    outputs = _to_class(outputs)

    if transform:
        targets = _to_class(targets)

    # Flatten
    outputs = _flatten(outputs)
    targets = _flatten(targets)

    # Evaluate
    args = {
        'y_true': targets,
        'y_pred': outputs,
        'average': 'weighted',
        'zero_division': 0
    }

    p = precision_score(**args)
    r = recall_score(**args)

    return [p, r]


# Main

def evaluate(
    outputs_pth: str = 'outputs/',
    dataset_id: str = 'class',
    test_pth: str = 'test.json',
    model_id: str = 'densenet161',
    edges: bool = False,
    batch_size: int = 32,
    out_channels: int = 2,
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

    datasets = {
        'class': ClassDataset,
        'image': ImageDataset
    }

    testset = datasets.get(dataset_id, 'class')(
        json_pth=test_pth,
        modelname=model_id,
        edges=edges
    )

    loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    models = {
        'densenet161': DenseNet161,
        'small': SmallConvNet,
        'unet': UNet
    }

    inpt, _ = testset[0]

    in_channels = inpt.size()[0]

    model = models.get(model_id, 'densenet161')(in_channels, out_channels)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_pth, map_location=device))
    model.eval()

    # Evaluation
    metrics = {
        'pr': pr_eval
    }

    evaluate = metrics.get(metric_id, 'pr')

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

    stats_header = stats_headers.get(metric_id, 'pr')

    metric_mean = np.mean(values, axis=0)
    metric_std = np.std(values, axis=0)

    with open(stats_pth, 'w', newline='') as f:
        csv.writer(f).writerow(stats_header)
        csv.writer(f).writerow(list(np.concatenate((metric_mean, metric_std))))
