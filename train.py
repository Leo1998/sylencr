#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

import dataset
import losses
import models

def train_loop(dataloader, device, model, loss_fn, optimizer):
    num_batches = 0
    for X, X_norm, y, mask in dataloader:
        X, X_norm, y, mask = X.to(device), X_norm.to(device), y.to(device), mask.to(device)

        # Compute prediction and loss
        pred = model(X_norm)
        loss = loss_fn(pred, mask)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"batch: {num_batches}, loss: {loss:>7f}")
        num_batches += 1

def test_loop(dataloader, device, model, loss_fn):
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for X, X_norm, y, mask in dataloader:
            X, X_norm, y, mask = X.to(device), X_norm.to(device), y.to(device), mask.to(device)
            pred = model(X_norm)
            test_loss += loss_fn(pred, mask).item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    trainDataset = dataset.OriginalDataset('data/train/noise/keyboard', 'data/train/speech')
    print(f"Length of trainDataset: {len(trainDataset)}")
    testDataset = dataset.OriginalDataset('data/test/noise/keyboard', 'data/test/speech')
    print(f"Length of testDataset: {len(testDataset)}")

    windowedTrainDataset = dataset.WindowedDataset(trainDataset, 8)
    windowedTestDataset = dataset.WindowedDataset(testDataset, 8)

    batch_size = 128
    learning_rate = 1e-3 # default value for Adam
    epochs = 1

    num_workers = 4
    train_dataloader = DataLoader(windowedTrainDataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size)
    test_dataloader = DataLoader(windowedTestDataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size)

    device = 'cpu'
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = models.DnnModel().to(device)
    print(model)

    loss_fn = losses.CustomMaskMSE(alpha=8.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, device, model, loss_fn, optimizer)
        test_loop(test_dataloader, device, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights.pth')

