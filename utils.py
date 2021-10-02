#!/usr/bin/env python3
import random
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def to_stft(file, n_fft, sr, amplitude_mul=1.0, duration=None):
  if duration is None:
    y, _ = librosa.load(file, mono=True, sr=sr)
  else:
    y, _ = librosa.load(file, mono=True, sr=sr, offset=0, duration=duration)

  y = y * amplitude_mul

  D = librosa.stft(y, window='hann', n_fft=n_fft)
  return D.T

def stft_to_mel(D, n_fft, sr, n_mels):
  M = librosa.feature.melspectrogram(S=np.abs(D.T), n_fft=n_fft, n_mels=n_mels, sr=sr, power=1)
  return M.T

def mel_to_stft(M, n_fft, sr, phases):
  D = librosa.feature.inverse.mel_to_stft(M.T, n_fft=n_fft, sr=sr, power=1)
  return np.vectorize(cmath.rect)(D.T, phases)

def log_norm(S):
    return (librosa.amplitude_to_db(S, ref=np.max) + 80.0) / 80.0

def ilog_norm(S):
    return librosa.db_to_amplitude(S * 80.0 - 80.0, ref=1.0)

def pcen(S):
    pcen_S = librosa.pcen(S * (2**31), axis=0)
    return pcen_S

def write_stft_to_wav(file, D, sr):
  x = librosa.istft(D.T)

  sf.write(file, x, sr)

def plot_spectrum(tensor, db=False):
  if db:
    tensor = librosa.amplitude_to_db(tensor, ref=np.max)
  fig = plt.figure()
  axes = fig.add_subplot(111)
  im = axes.pcolormesh(tensor.T)
  fig.colorbar(im)
  plt.show()

# test
if __name__ == '__main__':
  sr = 11025
  D = to_stft(librosa.ex('trumpet'), 512, sr)
  print(f"D.shape: {D.shape}")
  M = stft_to_mel(D, 512, sr, 128)
  print(f"M.shape: {M.shape}")

  M_norm = log_norm(M)
  M = ilog_norm(M_norm)

  D_out = mel_to_stft(M, 512, sr, np.angle(D))
  write_stft_to_wav("out.wav", D_out, sr)