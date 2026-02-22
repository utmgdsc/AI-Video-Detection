"""
Audio preprocessing utilities.

Prepares audio for AASIST model.
"""

from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa


def load_audio(audio_path, sample_rate=16000):
    """
    Load audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (AASIST typically uses 16kHz)

    Returns:
        numpy array: Audio waveform
    """
    # TODO: Implement using librosa or torchaudio
    # Example with librosa:
    # import librosa
    # audio, sr = librosa.load(audio_path, sr=sample_rate)
    # return audio

    audio_path = str(audio_path)
    x, sr = sf.read(audio_path)

    # convert stereo -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # resample if needed
    if sr != sample_rate:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=sample_rate)

    return x.astype(np.float32)


def preprocess(audio_path):
    """
    Full preprocessing pipeline for audio.

    Args:
        audio_path: Path to audio file

    Returns:
        tensor: Preprocessed audio ready for model
    """
    # TODO: Implement based on AASIST requirements
    # 1. Load audio
    # 2. Resample if needed
    # 3. Normalize
    # 4. Convert to tensor

    x = load_audio(audio_path, sample_rate=16000)

    # normalization (safe, simple)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak

    # tensor shape [T] (your wrapper handles [T] -> [1,T])
    return torch.from_numpy(x)
