import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib

# Avoid Type 3 fonts in PDF output
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# File paths
clean_path = '/home/ali/EchoCrypt/new_dataset_phone/0/0.wav'
noisy_path = '/home/ali/EchoCrypt/noise_audio_dataset_phone/0/0.wav'

# Load audio
waveform_clean, sample_rate_clean = torchaudio.load(clean_path)
waveform_noisy, sample_rate_noisy = torchaudio.load(noisy_path)

# Use first channel if stereo
waveform_clean = waveform_clean[0:1, :]  # shape becomes [1, N]
waveform_noisy = waveform_noisy[0:1, :]

# MelSpectrogram transform
to_mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate_clean,
    n_mels=64,
    hop_length=300,
    n_fft=2048,
    win_length=1024
)

# Compute mel spectrograms
mel_clean = to_mel_spectrogram(waveform_clean)
mel_noisy = to_mel_spectrogram(waveform_noisy)

# Convert to decibels (log scale)
mel_clean_db = torchaudio.functional.amplitude_to_DB(mel_clean, multiplier=10, amin=1e-10, db_multiplier=0)
mel_noisy_db = torchaudio.functional.amplitude_to_DB(mel_noisy, multiplier=10, amin=1e-10, db_multiplier=0)

# Function to plot and save
def save_spectrogram(mel_db, title, filename):
    plt.figure(figsize=(6, 4))
    plt.imshow(mel_db.squeeze(0).numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Save both spectrograms
save_spectrogram(mel_clean_db, 'Mel Spectrogram - Clean Keystroke "0"', "spectrogram_clean.pdf")
save_spectrogram(mel_noisy_db, 'Mel Spectrogram - Noisy Keystroke "0"', "spectrogram_noisy.pdf")
