"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 2(b) - define each layer
        stride = (2,2)
        padding = "SAME"
        filter = 5
        pooler = 2
        self.conv1 = nn.Conv2d(3, 16, filter, stride, padding)
        self.pool = nn.MaxPool2d(pooler, stride, padding=0)
        self.conv2 = nn.Conv2d(16, 64, filter, stride, padding)
        self.conv3 = nn.Conv2d(64, 8, filter, stride, padding)
        self.fc_1 = nn.Linear(32, 2)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 2(b) - initialize the parameters for the convolutional layers
            nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1.0/(5*5*conv.in_channels)))
            pass

        # TODO: 2(b) - initialize the parameters for [self.fc_1]
        nn.init.normal_(conv.weight, mean=0.0, std=sqrt(1.0/(self.fc_1.in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # TODO: 2(b) - , forward pass
        activate = nn.ReLU(inplace=False)
        x = activate(self.conv1(x))
        x = self.pool(x)
        x = activate(self.conv2(x))
        x = self.pool(x)
        x = activate(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        return x
