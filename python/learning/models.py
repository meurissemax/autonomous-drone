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

class DenseNet(nn.Module):
    """
    Implementation of modified versions of DenseNet models.

    Last classification layer has been replaced by convolutions layers
    followed by 1 fully connected one.

    DenseNet models are used to predict class associated to input image.

    Input images must be in 320 x 180.

    Available DenseNet models are: '121' and '161' (values for 'densenet_id').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        densenet_id: str = '121'
    ):
        super().__init__()

        # Normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        )

        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        )

        # Pre trained original DenseNet
        self.densenet = torch.hub.load(
            'pytorch/vision:v0.8.1',
            f'densenet{densenet_id}',
            pretrained=True
        )

        # Remove last layer
        self.densenet = nn.Sequential(*list(self.densenet.features))

        # New layers
        self.first = Conv(in_channels, 3)

        self.convs = nn.ModuleList([
            Conv(1024, 128, kernel_size=5),
            Conv(128, 16)
        ])

        if densenet_id == '161':
            self.convs.insert(0, Conv(2208, 1024))

        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 1 * 6, out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensors) -> Tensors:
        x = self.first(x)

        x = (x - self.mean) / self.std
        x = self.densenet(x)

        for conv in self.convs:
            x = conv(x)

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

        # Flatten layer
        self.flatten = nn.Flatten()

        # Dense layers
        self.denses = nn.ModuleList(
            [
                Dense(7680, 4096),
                Dense(4096, 2048),
                Dense(2048, 128)
            ] + [Dense(128, 128) for i in range(5)]
        )

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
        x = self.flatten(x)

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


class MiDaS(nn.Module):
    """
    Implementation of the MiDaS network.

    This model is used to predict relative inverse depth of images.

    Input images must be in 384 x 224.

    Taken from:
        - https://pytorch.org/hub/intelisl_midas_v2/
    """

    def __init__(self, _in_channels: int, _out_channels: int):
        super().__init__()

        # Normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        )

        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        )

        # Load MiDaS model
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS')

    def forward(self, x: Tensors) -> Tensors:
        x = (x - self.mean) / self.std
        x = self.midas(x)

        return x
