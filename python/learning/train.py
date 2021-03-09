#!/usr/bin/env python

"""
Implementation of the training procedure of the deep learning models.
"""

###########
# Imports #
###########

import csv
import numpy as np
import os
import sys
import torch
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ClassDataset, ImageDataset
from models import DenseNet161, SmallConvNet, UNet

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from misc.plots import plt  # noqa: E402


#############
# Functions #
#############

def train_epoch(
    loader: DataLoader,
    device: torch.device,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer
) -> list:
    """
    Train a model for one epoch.
    """

    losses = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


########
# Main #
########

def main(
    outputs_pth: str = 'outputs/',
    criterion_id: str = 'mse',
    dataset_id: str = 'class',
    train_pth: str = 'train.json',
    model_id: str = 'densenet161',
    augment: bool = False,
    edges: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    out_channels: int = 2,
    num_epochs: int = 20
):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f'Device: {device}')

    # Output folder
    os.makedirs(os.path.dirname(outputs_pth), exist_ok=True)

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder_pth = os.path.join(outputs_pth, now)
    os.makedirs(folder_pth, exist_ok=True)

    # Criterion and target dtype
    criterions = {
        'mse': (nn.MSELoss(), torch.float, 'MSE Loss'),
        'nll': (nn.NLLLoss(), torch.long, 'NLL Loss')
    }

    criterion, dtype, ylabel = criterions.get(criterion_id)

    # Data set and data loader
    print('Loading data set...')

    datasets = {
        'class': ClassDataset,
        'image': ImageDataset
    }

    trainset = datasets.get(dataset_id)(
        json_pth=train_pth,
        modelname=model_id,
        augment=augment,
        dtype=dtype,
        edges=edges
    )

    loader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    models = {
        'densenet161': DenseNet161,
        'small': SmallConvNet,
        'unet': UNet
    }

    inpt, _ = trainset[0]

    in_channels = inpt.size()[0]

    model = models.get(model_id)(in_channels, out_channels)
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = Adam(model.parameters())

    # Statistics file
    stats_pth = os.path.join(folder_pth, 'train.csv')

    with open(stats_pth, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch',
            'train_loss_mean',
            'train_loss_std'
        ])

    # Training
    epochs = range(num_epochs)
    mean_losses = []

    for epoch in tqdm(epochs):
        train_losses = train_epoch(loader, device, model, criterion, optimizer)

        # Statistics
        mean_losses.append(np.mean(train_losses))

        with open(stats_pth, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                np.mean(train_losses),
                np.std(train_losses)
            ])

        # Save weights
        if epoch == epochs[-1]:
            model_name = os.path.join(
                folder_pth,
                f'{model.__class__.__name__.lower()}.pth'
            )
            torch.save(model.state_dict(), model_name)

    # Plot
    plt.plot(range(1, num_epochs + 1), mean_losses)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    plt.savefig(os.path.join(folder_pth, 'train.pdf'))
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a deep learning model.'
    )

    parser.add_argument(
        '-o',
        '--outputs',
        type=str,
        default='outputs/',
        help='path to outputs folder'
    )

    parser.add_argument(
        '-c',
        '--criterion',
        type=str,
        default='mse',
        choices=['mse', 'nll'],
        help='criterion to use'
    )

    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='class',
        choices=['class', 'image'],
        help='data set to use'
    )

    parser.add_argument(
        '-t',
        '--train',
        type=str,
        default='train.json',
        help='path to JSON file with training data'
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='densenet161',
        choices=['densenet161', 'small', 'unet'],
        help='model to train'
    )

    parser.add_argument(
        '-a',
        '--augment',
        default=False,
        action='store_true',
        help='flag to enable data augmentation'
    )

    parser.add_argument(
        '-e',
        '--edges',
        default=False,
        action='store_true',
        help='flag to work with edges'
    )

    parser.add_argument(
        '-b',
        '--batch',
        type=int,
        default=32,
        help='batch size'
    )

    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=0,
        help='number of workers'
    )

    parser.add_argument(
        '-n',
        '--channels',
        type=int,
        default=2,
        help='number output channels'
    )

    parser.add_argument(
        '-p',
        '--epochs',
        type=int,
        default=20,
        help='number of epochs'
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
        num_workers=args.workers,
        out_channels=args.channels,
        num_epochs=args.epochs
    )
