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


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        #inputs is (8, 128, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (8,10), (1,1), padding=(4,5), padding_mode='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (4,10), (2,1), padding=(2,5), padding_mode='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, (2,10), (2,1), padding=(1,5), padding_mode='zeros'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, (2,10), (2,1), padding=(1,5), padding_mode='zeros'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_block1 = nn.Sequential(
            nn.Linear(4224, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
        )
        self.linear_out = nn.Sequential(
            nn.Linear(500, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear_relu_block1(x)
        x = self.linear_out(x)
        return x