#!/usr/bin/env python3
import torch
import utils
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
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
        M_noise = utils.stft_to_mel(D_noise, self.n_fft, self.sr, self.n_mels)
        #print(f"M_noise.shape: {M_noise.shape}")

        D_speech = utils.to_stft(speech_file, self.n_fft, self.sr)
        #print(f"D_speech.shape: {D_speech.shape}")
        M_speech = utils.stft_to_mel(D_speech, self.n_fft, self.sr, self.n_mels)
        #print(f"M_speech.shape: {M_speech.shape}")

        M_noise_fitted = M_noise
        while M_noise_fitted.shape[0] < M_speech.shape[0]:
            M_noise_fitted = np.append(M_noise_fitted, M_noise, axis = 0)
        M_noise_fitted = M_noise_fitted[:M_speech.shape[0],:]
        #print(f"M_noise_fitted.shape: {M_noise_fitted.shape}")

        M_mixed = M_speech + M_noise_fitted

        X = M_mixed
        y = M_speech

        return X, y

if __name__ == '__main__':
    dataset = Dataset('data/noise', 'data/speech')
    print(f"Length of Dataset: {len(dataset)}")
    for X, y in dataset:
        pass