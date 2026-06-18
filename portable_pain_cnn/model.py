#!/usr/bin/env python3
"""
Model definition for the Pain Spectrogram CNN.

This is the single source of truth for the SimpleCNN architecture. The trained
weights in `cnn_model.pt` were produced by experiments/05-spectrogram_cnn and
match this module exactly. `predict.py` imports SimpleCNN from here.

Architecture (617,921 parameters):
    Input  : (N, 1, 128, 300)   # 1-channel mel-spectrogram, 128 mels x 300 frames
    Conv1  : Conv2d(1->32, 3x3) + BN + ReLU + MaxPool(2,2)
    Conv2  : Conv2d(32->64, 3x3) + BN + ReLU + MaxPool(2,2)
    Conv3  : Conv2d(64->128, 3x3) + BN + ReLU + AdaptiveAvgPool(4,4)
    FC1    : Linear(2048 -> 256) + ReLU + Dropout(0.5)
    FC2    : Linear(256 -> 1)
    Output : scalar pain score per sample (clipped to [1, 10] by the caller)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128
MAX_FRAMES = 300


class SimpleCNN(nn.Module):
    """Lightweight CNN for mel-spectrogram -> pain-score regression."""

    def __init__(self, n_mels: int = N_MELS, max_length: int = MAX_FRAMES):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected
        self.flattened_size = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()
