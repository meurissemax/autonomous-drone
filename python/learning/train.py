#!/usr/bin/env python

"""
Implementation of the training procedure of the deep
learning models.
"""

###########
# Imports #
###########

import csv
import numpy as np
import os
import torch
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IndoorDataset
from models import DenseNet161


#############
# Functions #
#############

def train_epoch(loader, device, model, criterion, optimizer):
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
    outputs_pth='outputs/',
    criterion_id='mse',
    train_pth='train.json',
    augment=False,
    batch_size=32,
    num_workers=0,
    model_id='densenet161',
    num_epochs=20
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
        'mse': (nn.MSELoss(), torch.float)
    }

    criterion, dtype = criterions.get(criterion_id)

    # Data set and data loader
    print('Loading data set...')

    trainset = IndoorDataset(train_pth, model_id, augment, dtype)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Model
    models = {
        'densenet161': DenseNet161
    }

    inpt, trgt = trainset[0]

    in_channels = inpt.size()[0]
    out_channels = trgt.size()[0]

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

    for epoch in tqdm(epochs):
        train_losses = train_epoch(loader, device, model, criterion, optimizer)

        # Statistics
        with open(stats_pth, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                np.mean(train_losses),
                np.std(train_losses)
            ])

        # Save weights
        if epoch == epochs[-1]:
            model_name = os.path.join(folder_pth, f'{model.__class__.__name__.lower()}.pth')
            torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a deep learning model.')

    parser.add_argument('-o', '--outputs', type=str, default='outputs/', help='path to outputs folder')
    parser.add_argument('-c', '--criterion', type=str, default='mse', choices=['mse'], help='criterion to use')
    parser.add_argument('-t', '--train', type=str, default='train.json', help='path to JSON file with training data')
    parser.add_argument('-a', '--augment', default=False, action='store_true', help='flag to enable data augmentation')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-w', '--workers', type=int, default=0, help='number of workers')
    parser.add_argument('-m', '--model', type=str, default='densenet161', choices=['densenet161'], help='model to train')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs')

    args = parser.parse_args()

    main(
        outputs_pth=args.outputs,
        criterion_id=args.criterion,
        train_pth=args.train,
        augment=args.augment,
        batch_size=args.batch,
        num_workers=args.workers,
        model_id=args.model,
        num_epochs=args.epochs
    )
