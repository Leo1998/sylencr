#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == '__main__':
    dataset = dataset.Dataset('data/noise', 'data/speech')
    print(f"Length of Dataset: {len(dataset)}")

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    '''for D_mixed, M_mixed, D_speech, M_speech, D_noise, M_noise in dataset:
        utils.plot_spectrum(M_mixed)
        utils.plot_spectrum(utils.log_norm(M_mixed))'''