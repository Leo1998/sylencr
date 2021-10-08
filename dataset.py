#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import RandomSampler

import utils

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, noise_path, speech_path):
        self.n_fft = 512
        self.n_mels = 128
        self.sr = 11025
        self.noise_path = noise_path
        self.speech_path = speech_path

        self.noise_files = []
        self.speech_files = []
        for (path, dirs, files) in os.walk(noise_path):
            for file in files:
                file = os.path.join(path, file)
                if not file.lower().endswith('.wav'):
                    continue
                #print(f"Adding noise file: {file}")
                self.noise_files.append(file)
        
        for (path, dirs, files) in os.walk(self.speech_path):
            for file in files:
                file = os.path.join(path, file)
                if not file.lower().endswith('.wav'):
                    continue
                #print(f"Adding speech file: {file}")
                self.speech_files.append(file)

    def __len__(self):
        return len(self.noise_files) * len(self.speech_files)

    def __getitem__(self, index):
        noise_file = self.noise_files[index % len(self.noise_files)]
        speech_file = self.speech_files[int(index / len(self.noise_files))]

        #print(f"Generating mix of speech file: {speech_file} and noise file: {noise_file}")

        D_noise = utils.to_stft(noise_file, self.n_fft, self.sr)
        #print(f"D_noise.shape: {D_noise.shape}")
        D_speech = utils.to_stft(speech_file, self.n_fft, self.sr)
        #print(f"D_speech.shape: {D_speech.shape}")
        
        D_noise_orig = D_noise
        while D_noise.shape[0] < D_speech.shape[0]:
            D_noise = np.append(D_noise, D_noise, axis = 0)
        D_noise = D_noise[:D_speech.shape[0],:]

        mix = 0.5
        D_mixed = mix * D_speech + (1/mix) * D_noise

        M_noise = utils.stft_to_mel(D_noise, self.n_fft, self.sr, self.n_mels)
        #print(f"M_noise.shape: {M_noise.shape}")
        M_speech = utils.stft_to_mel(D_speech, self.n_fft, self.sr, self.n_mels)
        #print(f"M_speech.shape: {M_speech.shape}")

        M_mixed = utils.stft_to_mel(D_mixed, self.n_fft, self.sr, self.n_mels)

        return D_mixed, M_mixed, D_speech, M_speech, D_noise, M_noise


class WindowedTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, trainDataset, n_time_windows):
        self.trainDataset = trainDataset
        self.n_time_windows = n_time_windows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise ValueError('No multiprocessing supported yet!')

        sampler = RandomSampler(self.trainDataset)
        
        for index in sampler:
            D_mixed, M_mixed, D_speech, M_speech, D_noise, M_noise = self.trainDataset[index]
            for i in range(self.n_time_windows, M_mixed.shape[0]):
                X = M_mixed[i-self.n_time_windows:i]
                X_norm = utils.log_norm(X)
                y = M_speech[i-1]
                mask = utils.magnitude_mask(X[-1], y)

                yield X, X_norm, y, mask

