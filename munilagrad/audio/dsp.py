import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def frames(y,frame_length = 2048, hop_length = 512):
    y = np.asarray(y)
    num_frames = 1 + (len(y) - frame_length) // hop_length
    base_indices = np.arange(frame_length)
    offsets = np.arange(num_frames) * hop_length
    frame_indices = offsets[:,None] + base_indices
    frames = y[frame_indices]
    return frames

def hann_window(window_length):
    n = np.arange(window_length)
    window = 0.5 * (1 - np.cos((2 * np.pi * n) / (window_length - 1)))
    return window

def stft(y,n_fft=2048,hop_len=512):
    frame = frames(y,frame_length=n_fft,hop_length=hop_len)
    window = hann_window(n_fft)
    windowed_frame = frame * window
    complex_stft = np.fft.rfft(windowed_frame,axis=-1)
    mag_spectograme = np.abs(complex_stft)
    return mag_spectograme.T

def mel_spectograme(y,sr,n_fft=2048,hop_len=512,n_mels=3):
    mag_spec = stft(y, n_fft, hop_len)
    power_spec = mag_spec ** 2
    bins = (n_fft // 2) + 1  #nbins are 1025
    freq_resol = n_fft // 2
    min_hz = 0
    max_hz = sr /2 
    min_mels = 0.0
    max_mel = 2595.0 * np.log10(1.0 + (max_hz / 700.0))
    mel_points = np.linspace(min_mels,max_mel,n_mels+2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    f_bank = np.zeros((n_mels, (n_fft // 2) + 1))
    for i in range(1,n_mels+1):
        left_bin = bin_indices[i-1]
        peak_bin = bin_indices[i]
        right_bin = bin_indices[i+1]
        
        if peak_bin > left_bin:
            for k in range(left_bin,peak_bin):
                f_bank[i-1,k] = (k-left_bin) / (peak_bin-left_bin)
        
        f_bank[i-1,peak_bin] = 1.0
        
        if right_bin > peak_bin:
            for k in range(peak_bin+1,right_bin+1):
                f_bank[i-1,k] = (right_bin - k) / (right_bin-peak_bin)
    mel_spec = np.dot(f_bank,power_spec)
    log_mel_spec = 10.0 * np.log10(mel_spec + 1e-9) 
    return log_mel_spec
