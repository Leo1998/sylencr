#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

import dataset
import models

def train_loop(dataloader, device, model, loss_fn, optimizer):
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

    model = models.DnnModel().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, device, model, loss_fn, optimizer)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights.pth')

