#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import utils
import dataset

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


def train_loop(dataloader, device, model, loss_fn, optimizer):
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"loss: {loss:>7f}")

'''def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")'''

if __name__ == '__main__':
    trainDataset = dataset.TrainDataset('data/noise', 'data/speech')
    print(f"Length of trainDataset: {len(trainDataset)}")

    windowedDataset = dataset.WindowedTrainDataset(trainDataset, 8)

    batch_size = 64
    learning_rate = 1e-3
    epochs = 1

    train_dataloader = DataLoader(windowedDataset, num_workers=0, batch_size=batch_size)

    device = 'cpu'
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = DnnModel().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, device, model, loss_fn, optimizer)
    print("Done!")

