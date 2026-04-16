import numpy as np

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