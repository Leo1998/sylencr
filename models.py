#!/usr/bin/env python3

from torch import nn

class DnnModel(nn.Module):
    def __init__(self):
        super(DnnModel, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_block1 = nn.Sequential(
            nn.Linear(8 * 128, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        self.linear_relu_block2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        self.linear_out = nn.Sequential(
            nn.Linear(500, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_block1(x)
        x = self.linear_relu_block2(x)
        x = self.linear_out(x)
        return x