"""
Implementation of the deep learning models.
"""

###########
# Imports #
###########

import torch
import torch.nn as nn

from typing import Iterable


##########
# Typing #
##########

Tensors = Iterable[torch.Tensor]


###########
# Classes #
###########

# Generic classes

class Conv(nn.Sequential):
    """
    Implementation of a generic convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConv(nn.Sequential):
    """
    Implementation of a generic double convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            Conv(out_channels, out_channels, kernel_size, stride, padding)
        )


class Dense(nn.Sequential):
    """
    Implementation of a generic dense layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )


# Models

class DenseNet161(nn.Module):
    """
    Implementation of the modified version of model DenseNet161.

    Last classification layer has been replaced by 3 convolution layers
    followed by 1 fully connected one.

    This model is used to predict class associated to input image.

    Input images must be in 320 x 180.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Pre trained original DenseNet161
        self.densenet161 = torch.hub.load(
            'pytorch/vision:v0.6.0',
            'densenet161',
            pretrained=True
        )

        # Remove last layer
        self.densenet161 = nn.Sequential(*list(self.densenet161.features))

        # New layers
        self.first = Conv(in_channels, 3)

        self.conv1 = Conv(2208, 1024)
        self.conv2 = Conv(1024, 128, kernel_size=5)
        self.conv3 = Conv(128, 16)

        self.last = nn.Sequential(
            nn.Linear(16 * 1 * 6, out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensors) -> Tensors:
        x = self.first(x)

        x = self.densenet161(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.last(x)

        return x


class SmallConvNet(nn.Module):
    """
    Implementation of a small convolution network for image classification.

    This model is used to predict class associated to input image.

    Input images must be in 320 x 180.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First layer
        self.first = Conv(in_channels, 3)

        # Convolutional layers
        self.convs = nn.ModuleList([
            DoubleConv(3, 32),
            DoubleConv(32, 32),
            DoubleConv(32, 64),
            DoubleConv(64, 64),
            DoubleConv(64, 128),
        ])

        # Max pool layer
        self.max_pool = nn.MaxPool2d(2, ceil_mode=True)

        # Drop out layer
        self.drop_out = nn.Dropout(p=0.8)

        # Dense layers
        self.denses = nn.ModuleList([
            Dense(7680, 4096),
            Dense(4096, 2048),
            Dense(2048, 1024),
            Dense(1024, 512),
            Dense(512, 128)
        ])

        # Last layers
        self.last = nn.Sequential(
            nn.Linear(128, out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensors) -> Tensors:
        x = self.first(x)

        for conv in self.convs:
            x = conv(x)
            x = self.max_pool(x)

        x = self.drop_out(x)
        x = torch.flatten(x, start_dim=1)

        for dense in self.denses:
            x = dense(x)

        x = self.last(x)

        return x


class UNet(nn.Module):
    """
    Implementation of the U-Net network.

    This model is used to predict mask associated to input image.

    Inspired from:
        - https://github.com/francois-rozet/adopptrs/
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        depth = 4

        self.downs = nn.ModuleList(
            [DoubleConv(in_channels, 64)] + [
                DoubleConv(64 * (2 ** i), 128 * (2 ** i))
                for i in range(depth)
            ]
        )

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        self.ups = nn.ModuleList([
            DoubleConv((64 + 128) * (2 ** i), 64 * (2 ** i))
            for i in reversed(range(depth))
        ])

        self.last = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: Tensors) -> Tensors:
        features, shapes = [], []

        # Downhill
        for down in self.downs[:-1]:
            x = down(x)
            features.append(x)
            shapes.append(x.shape[-2:])
            x = self.maxpool(x)

        x = self.downs[-1](x)

        # Uphill
        for up in self.ups:
            x = self.upsample(x)
            x = torch.cat([
                x[:, :, :shapes[-1][0], :shapes.pop()[1]],
                features.pop()
            ], dim=1)
            x = up(x)

        x = self.last(x)

        return x
