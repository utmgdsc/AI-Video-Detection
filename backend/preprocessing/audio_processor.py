"""
Audio preprocessing utilities.

Prepares audio for AASIST model.
"""

import numpy as np


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

    raise NotImplementedError("Implement load_audio()")


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

    raise NotImplementedError("Implement preprocess()")
