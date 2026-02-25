"""Audio processing utilities for chord recognition."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        mono: Convert to mono if True.
        normalize: Normalize audio if True.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        audio, sr = librosa.load(
            file_path, 
            sr=sample_rate, 
            mono=mono,
            res_type="kaiser_fast"
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 22050,
    format: str = "wav",
) -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array to save.
        file_path: Output file path.
        sample_rate: Sample rate.
        format: Audio format.
    """
    try:
        sf.write(file_path, audio, sample_rate, format=format)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {file_path}: {e}")


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment_length: float = 3.0,
    overlap: float = 0.5,
) -> List[np.ndarray]:
    """Segment audio into overlapping chunks.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate.
        segment_length: Length of each segment in seconds.
        overlap: Overlap ratio between segments.
        
    Returns:
        List of audio segments.
    """
    segment_samples = int(segment_length * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        end = start + segment_samples
        segments.append(audio[start:end])
    
    return segments


def pad_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = "constant",
    constant_values: float = 0.0,
) -> np.ndarray:
    """Pad audio to target length.
    
    Args:
        audio: Input audio array.
        target_length: Target length in samples.
        mode: Padding mode.
        constant_values: Constant value for padding.
        
    Returns:
        Padded audio array.
    """
    if len(audio) >= target_length:
        return audio[:target_length]
    
    pad_width = target_length - len(audio)
    if mode == "constant":
        return np.pad(audio, (0, pad_width), mode=mode, constant_values=constant_values)
    else:
        return np.pad(audio, (0, pad_width), mode=mode)


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio.
    
    Args:
        audio: Input audio array.
        coeff: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized audio array.
    """
    return librosa.effects.preemphasis(audio, coef=coeff)


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Detect silent frames in audio.
    
    Args:
        audio: Input audio array.
        threshold: Silence threshold.
        frame_length: Frame length for analysis.
        hop_length: Hop length for analysis.
        
    Returns:
        Boolean array indicating silent frames.
    """
    # Calculate RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    return rms < threshold


def remove_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove silent regions from audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate.
        threshold: Silence threshold.
        frame_length: Frame length for analysis.
        hop_length: Hop length for analysis.
        
    Returns:
        Audio with silence removed.
    """
    silent_frames = detect_silence(audio, threshold, frame_length, hop_length)
    
    # Convert frame indices to sample indices
    silent_samples = []
    for i, is_silent in enumerate(silent_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(audio))
        if is_silent:
            silent_samples.extend(range(start_sample, end_sample))
    
    # Keep non-silent samples
    mask = np.ones(len(audio), dtype=bool)
    mask[silent_samples] = False
    
    return audio[mask]


def get_audio_info(file_path: Union[str, Path]) -> dict:
    """Get information about an audio file.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dictionary with audio file information.
    """
    try:
        info = sf.info(file_path)
        return {
            "file_path": str(file_path),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get audio info for {file_path}: {e}")


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid audio file.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        True if valid audio file, False otherwise.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False
        
        # Try to load with librosa
        librosa.load(file_path, sr=None, duration=0.1)
        return True
    except Exception:
        return False
