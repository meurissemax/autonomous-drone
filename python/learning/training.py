"""
Implementation of the training procedure of the deep learning models.
"""

###########
# Imports #
###########

import csv
import numpy as np
import os
import torch
import torch.nn as nn

from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import ClassDataset, ImageDataset
from .models import DenseNet161, SmallConvNet, UNet
from plots.latex import plt


#############
# Functions #
#############

# Utilities

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


# Main

def train(
    outputs_pth: str = 'outputs/',
    criterion_id: str = 'mse',
    dataset_id: str = 'class',
    train_pth: str = 'train.json',
    model_id: str = 'densenet161',
    augment: bool = False,
    edges: bool = False,
    batch_size: int = 32,
    out_channels: int = 2,
    num_epochs: int = 30,
    weights_pth: str = 'weights.pth'
):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f'Device: {device}')

    # Output folder
    os.makedirs(os.path.dirname(outputs_pth), exist_ok=True)

    # Criterion and target dtype
    criterions = {
        'mse': (nn.MSELoss(), torch.float, 'MSE Loss'),
        'nll': (nn.NLLLoss(), torch.long, 'NLL Loss')
    }

    criterion, dtype, ylabel = criterions.get(criterion_id, 'mse')

    # Data set and data loader
    print('Loading data set...')

    datasets = {
        'class': ClassDataset,
        'image': ImageDataset
    }

    trainset = datasets.get(dataset_id, 'class')(
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

    model = models.get(model_id, 'densenet161')(in_channels, out_channels)
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = Adam(model.parameters())

    # Statistics file
    stats_pth = os.path.join(outputs_pth, 'train.csv')

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
            torch.save(model.state_dict(), weights_pth)

    # Plot
    plt.plot(range(1, num_epochs + 1), mean_losses)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    plt.savefig(os.path.join(outputs_pth, 'train.pdf'))
    plt.close()
