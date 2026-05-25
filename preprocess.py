import librosa
import numpy as np
from config import MAX_LEN, N_MELS


def audio_to_melspec(y, sr, max_len=MAX_LEN):
    """Convert raw audio to a fixed-size mel-spectrogram (dB scale)."""
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db


def augment_audio(y, sr):
    """Return 3 augmented versions of the audio clip."""
    augmented = []

    # 1. add random noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)

    # 2. pitch shift up slightly
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))

    # 3. time stretch slightly
    augmented.append(librosa.effects.time_stretch(y, rate=0.9))

    return augmented