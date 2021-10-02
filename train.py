#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == '__main__':
    trainDataset = dataset.TrainDataset('data/noise', 'data/speech')
    print(f"Length of trainDataset: {len(trainDataset)}")

    windowedDataset = dataset.WindowedTrainDataset(trainDataset, 8)

    train_dataloader = DataLoader(windowedDataset, num_workers=0, batch_size=64)

    X, y = next(iter(train_dataloader))
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")


