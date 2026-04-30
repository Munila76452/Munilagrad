import librosa
import librosa.display as display
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

class DFTbase(nn.Module):
    def __init__(self):
        super(DFTbase, self).__init__()
    
    def dft_matrices(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n) 
        w = np.power(omega, x * y)
        return w 
        
    def idft_matrices(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        w = np.power(omega, x * y)
        return w 
    
class STFT(DFTbase):
    def __init__(self, n_fft=2048, hop_len=None, win_len=None, window='hann', center=True,
                 pad_mode='reflect', freeze_parameter=True):
        super(STFT, self).__init__() 
        
        assert pad_mode in ['constant', 'reflect']
        
        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        
        if win_len is None:
            win_len = n_fft
            
        if hop_len is None:
            hop_len = int(win_len // 4)
            
        fft_window = librosa.filters.get_window(window, win_len, fftbins=True)
        fft_window = librosa.util.pad_center(fft_window, n_fft)
        
        self.W = self.dft_matrices(n_fft)
        out_channel = n_fft // 2 + 1
        
        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=n_fft,
                                   stride=hop_len, padding=0, dilation=1, groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=n_fft,
                                   stride=hop_len, padding=0, dilation=1, groups=1, bias=False)
        
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channel] * fft_window[:, None]).T)[:, None, :]
        
        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channel] * fft_window[:, None]).T)[:, None, :]
        
        if freeze_parameter:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, input):
        """input: (batch_size, data_length)
        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """
        x = input[:, None, :]
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)

        return real, imag
    
class spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_len=None, win_len=None, window='hann', center=True,
                 pad_mode='reflect', power=2.0, freeze_parameter=True):
        
        super(spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_len=hop_len, 
            win_len=win_len, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameter=freeze_parameter)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """
        (real, imag) = self.stft.forward(input)
        spectrogram = real ** 2 + imag ** 2
    
        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)
            
        return spectrogram

class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
                 ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.melW = nn.Parameter(torch.Tensor(melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, 1, time_steps, freq_bins)"""
        # Collapse the 1-channel dimension for matrix multiplication
        input = input.squeeze(1) 
        
        # Linear to Mel
        mel_spectrogram = torch.matmul(input, self.melW)

        # Log compression
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            # Dynamic range compression
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec
    
    