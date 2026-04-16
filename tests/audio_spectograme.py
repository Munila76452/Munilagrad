from munilagrad.audio import dsp
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal

def record_and_compare(duration=4, sample_rate=16000):
    print(f"RECORDING FOR {duration} SECONDS...")
    print("Speak into your Mac microphone now!")
    
    # 1. Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait() 
    print("Recording Complete!\n")
    
    audio_array = recording.flatten()
    audio_array = audio_array / np.max(np.abs(audio_array))

    max_time = len(audio_array) / sample_rate
    max_freq = sample_rate / 2  # Nyquist limit (8000 Hz)
    extent = [0, max_time, 0, max_freq]

    print(" Running SciPy STFT...")
    # SciPy uses 'noverlap', which is frame_length - hop_length (2048 - 512 = 1536)
    freqs, times, scipy_mag = signal.spectrogram(
        audio_array, 
        fs=sample_rate, 
        window='hann', 
        nperseg=2048, 
        noverlap=1536,
        mode='magnitude' # We force it to magnitude so it matches your math!
    )
    scipy_log = 10 * np.log10(scipy_mag**2 + 1e-10)

    print(" Running Munilagrad STFT...")
    custom_mag = dsp.stft(audio_array, n_fft=2048, hop_len=512)
    custom_log = 10 * np.log10(custom_mag**2 + 1e-10)
    custom_log = custom_log - np.max(custom_log)    

    print(" Plotting comparison...")
    # Create a figure with 2 rows and 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top Plot (SciPy) - Let's lock it so we can compare fairly
    im1 = ax1.imshow(scipy_log, aspect='auto', origin='lower', cmap='magma', extent=extent, 
                     vmin=np.max(scipy_log) - 80, vmax=np.max(scipy_log))
    ax1.set_title("Industry Standard: SciPy 'signal.spectrogram'")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(0, 8000)
    fig.colorbar(im1, ax=ax1, format='%+2.0f dB')

     # Bottom Plot (Munilagrad) - Lock the colors to an 80 dB range!
    im2 = ax2.imshow(custom_log, aspect='auto', origin='lower', cmap='magma', extent=extent, 
                     vmin=-80, vmax=0)
    ax2.set_title("Your Custom Math: Munilagrad Engine")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_ylim(0, 8000)
    fig.colorbar(im2, ax=ax2, format='%+2.0f dB')

    # Add a little spacing between the charts
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    record_and_compare(duration=4)