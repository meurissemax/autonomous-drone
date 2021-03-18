"""
Implementation of data sets used by the deep learning models.

Each data set contains input images and associated targets. Targets depend
on the data set (a class for the ClassDataset and an image for the
ImageDataset).
"""

###########
# Imports #
###########

import cv2
import io
import json
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple


##########
# Typing #
##########

PILImage = Image.Image


#############
# Functions #
#############

def to_edges(img: PILImage) -> PILImage:
    """
    Extract the edges from an image.
    """

    # Convert to numpy array
    img = np.array(img)

    # Extract edges
    lo_thresh = 50
    hi_thresh = 250
    filter_size = 3

    img = cv2.Canny(
        img,
        lo_thresh,
        hi_thresh,
        apertureSize=filter_size,
        L2gradient=True
    )

    # Convert to PIL Image
    img = Image.fromarray(img)

    return img


###########
# Classes #
###########

class ClassDataset(Dataset):
    """
    Data set that contains input images and associated target classes.

    The JSON data file has to have the following format (example for a
    4 classes classification problem):

    [
        {
            "image": "path/to/image.png",
            "target": [0, 0, 1, 0]
        },
        ...
    ]

    Each pair is composed of the path to the image and the target whose
    format is the target tensor used by PyTorch.
    """

    def __init__(
        self,
        json_pth: str,
        augment: bool = False,
        dtype: torch.dtype = torch.float,
        edges: bool = False
    ):
        super().__init__()

        # Data
        self.data = []

        # Process
        self.process = transforms.ToTensor()

        # Data augmentation
        if augment:
            self.augment = transforms.RandomChoice([
                lambda x: x,
                lambda x: x.filter(ImageFilter.BLUR),
                lambda x: x.filter(ImageFilter.EDGE_ENHANCE),
                lambda x: x.filter(ImageFilter.SMOOTH),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=(0.2, 0.6),
                    saturation=(0.2, 0.6)
                )
            ])
        else:
            self.augment = None

        # Target data type
        self.dtype = dtype

        # Edges
        self.edges = edges

        # Get data
        with open(json_pth, 'r') as json_file:
            pairs = json.load(json_file)

            for pair in tqdm(pairs):
                data = []

                # Image
                with open(pair['image'], 'rb') as img:
                    img_bytes = io.BytesIO(img.read())
                    data.append(img_bytes)

                # Target
                target = pair['target']
                data.append(target)

                # Save tuple
                self.data.append(tuple(data))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get data
        img, target = self.data[index]

        # Image
        img = Image.open(img)

        if self.edges:
            img = to_edges(img)
        elif self.augment is not None:
            img = self.augment(img)

        img = self.process(img)

        # Target
        target = torch.tensor(target, dtype=self.dtype)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class ImageDataset(Dataset):
    """
    Data set that contains input images and associated target images.

    The JSON data file has to have the following format:

    [
        {
            "image": "path/to/image.png",
            "target": "path/to/target.png"
        },
        ...
    ]

    Each pair is composed of the path to the input image and the path to the
    target image.
    """

    def __init__(
        self,
        json_pth: str,
        augment: bool = False,
        dtype: torch.dtype = torch.long,
        edges: bool = False
    ):
        super().__init__()

        # Data
        self.data = []

        # Process
        self.process = transforms.ToTensor()

        # Data augmentation
        if augment:
            self.augment = transforms.RandomChoice([
                lambda x: x,
                lambda x: x.filter(ImageFilter.BLUR),
                lambda x: x.filter(ImageFilter.EDGE_ENHANCE),
                lambda x: x.filter(ImageFilter.SMOOTH),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=(0.2, 0.6),
                    saturation=(0.2, 0.6)
                )
            ])
        else:
            self.augment = None

        # Target data type
        self.dtype = dtype

        # Edges
        self.edges = edges

        # Get data
        with open(json_pth, 'r') as json_file:
            pairs = json.load(json_file)

            for pair in tqdm(pairs):
                data = []

                # Input
                with open(pair['image'], 'rb') as img:
                    img_bytes = io.BytesIO(img.read())
                    data.append(img_bytes)

                # Target
                with open(pair['target'], 'rb') as img:
                    img_bytes = io.BytesIO(img.read())
                    data.append(img_bytes)

                # Save tuple
                self.data.append(tuple(data))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get data
        inpt, trgt = self.data[index]

        # Input
        inpt = Image.open(inpt)

        if self.edges:
            img = to_edges(img)
        elif self.augment is not None:
            inpt = self.augment(inpt)

        inpt = self.process(inpt)

        # Target
        trgt = Image.open(trgt)
        trgt = np.array(trgt, dtype='uint8', copy=True)
        trgt = torch.from_numpy(trgt)
        trgt = trgt.to(dtype=self.dtype)
        trgt = trgt.squeeze(0)

        return inpt, trgt

    def __len__(self) -> int:
        return len(self.data)
