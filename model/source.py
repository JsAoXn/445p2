"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 3(a) - define each layer
        stride = 2
        padding = 2
        filter = 5
        pooler = 2
        self.conv1 = nn.Conv2d(3, 16, filter, stride, padding)
        self.pool = nn.MaxPool2d(pooler, stride, padding=0)
        self.conv2 = nn.Conv2d(16, 64, filter, stride, padding)
        self.conv3 = nn.Conv2d(64, 8, filter, stride, padding)
        self.fc1 = nn.Linear(32, 8)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 3(a) - initialize the parameters for the convolutional layers
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1.0/(5*5*conv.in_channels)))
        
        # TODO: 3(a) - initialize the parameters for [self.fc1]
        nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1.0/(self.fc1.in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        N, C, H, W = x.shape

        # TODO: 3(a) - forward pass
        activate = nn.ReLU(inplace=False)
        x = activate(self.conv1(x))
        x = self.pool(x)
        x = activate(self.conv2(x))
        x = self.pool(x)
        x = activate(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
