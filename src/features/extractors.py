"""Feature extraction for chord recognition."""

from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram


class ChromaExtractor:
    """Extract chroma features for chord recognition."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        method: str = "stft",
        bins_per_octave: int = 12,
        fmin: float = 65.41,
        fmax: Optional[float] = None,
        threshold: float = 0.0,
        norm: float = 2.0,
        win_len_smooth: int = 41,
        smoothing_window: str = "hann",
    ):
        """Initialize chroma extractor.
        
        Args:
            sample_rate: Sample rate of audio.
            hop_length: Hop length for STFT.
            n_fft: FFT window size.
            method: Chroma extraction method ('stft', 'cqt', 'cqt_harmonic').
            bins_per_octave: Number of bins per octave for CQT.
            fmin: Minimum frequency.
            fmax: Maximum frequency.
            threshold: Threshold for chroma features.
            norm: Normalization factor.
            win_len_smooth: Window length for smoothing.
            smoothing_window: Type of smoothing window.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.method = method
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.threshold = threshold
        self.norm = norm
        self.win_len_smooth = win_len_smooth
        self.smoothing_window = smoothing_window
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Chroma features array of shape (12, time_frames).
        """
        if self.method == "stft":
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                threshold=self.threshold,
                norm=self.norm,
                win_len_smooth=self.win_len_smooth,
                smoothing_window=self.smoothing_window,
            )
        elif self.method == "cqt":
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin,
                threshold=self.threshold,
                norm=self.norm,
            )
        elif self.method == "cqt_harmonic":
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin,
                threshold=self.threshold,
                norm=self.norm,
                harmonic=True,
            )
        else:
            raise ValueError(f"Unknown chroma method: {self.method}")
        
        return chroma
    
    def extract_mean(self, audio: np.ndarray) -> np.ndarray:
        """Extract mean chroma features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Mean chroma features array of shape (12,).
        """
        chroma = self.extract(audio)
        return np.mean(chroma, axis=1)


class MFCCExtractor:
    """Extract MFCC features for chord recognition."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        n_mfcc: int = 13,
        dct_type: int = 2,
        norm: str = "ortho",
        lifter: int = 0,
    ):
        """Initialize MFCC extractor.
        
        Args:
            sample_rate: Sample rate of audio.
            hop_length: Hop length for STFT.
            n_fft: FFT window size.
            n_mels: Number of mel bands.
            n_mfcc: Number of MFCC coefficients.
            dct_type: DCT type.
            norm: Normalization type.
            lifter: Liftering coefficient.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            MFCC features array of shape (n_mfcc, time_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            lifter=self.lifter,
        )
        return mfcc
    
    def extract_mean(self, audio: np.ndarray) -> np.ndarray:
        """Extract mean MFCC features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Mean MFCC features array of shape (n_mfcc,).
        """
        mfcc = self.extract(audio)
        return np.mean(mfcc, axis=1)


class SpectralExtractor:
    """Extract spectral features for chord recognition."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        include_centroid: bool = True,
        include_rolloff: bool = True,
        include_bandwidth: bool = True,
        include_zcr: bool = True,
    ):
        """Initialize spectral extractor.
        
        Args:
            sample_rate: Sample rate of audio.
            hop_length: Hop length for STFT.
            n_fft: FFT window size.
            include_centroid: Include spectral centroid.
            include_rolloff: Include spectral rolloff.
            include_bandwidth: Include spectral bandwidth.
            include_zcr: Include zero crossing rate.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.include_centroid = include_centroid
        self.include_rolloff = include_rolloff
        self.include_bandwidth = include_bandwidth
        self.include_zcr = include_zcr
    
    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Dictionary of spectral features.
        """
        features = {}
        
        if self.include_centroid:
            features["spectral_centroid"] = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, n_fft=self.n_fft
            )[0]
        
        if self.include_rolloff:
            features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, n_fft=self.n_fft
            )[0]
        
        if self.include_bandwidth:
            features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, n_fft=self.n_fft
            )[0]
        
        if self.include_zcr:
            features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
                audio, hop_length=self.hop_length
            )[0]
        
        return features
    
    def extract_mean(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract mean spectral features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Dictionary of mean spectral features.
        """
        features = self.extract(audio)
        return {k: np.mean(v) for k, v in features.items()}


class FeatureExtractor:
    """Main feature extractor combining multiple feature types."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        chroma_method: str = "stft",
        include_mfcc: bool = True,
        include_spectral: bool = True,
        **kwargs
    ):
        """Initialize feature extractor.
        
        Args:
            sample_rate: Sample rate of audio.
            hop_length: Hop length for STFT.
            n_fft: FFT window size.
            n_mels: Number of mel bands.
            chroma_method: Chroma extraction method.
            include_mfcc: Include MFCC features.
            include_spectral: Include spectral features.
            **kwargs: Additional arguments for feature extractors.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        
        # Initialize feature extractors
        self.chroma_extractor = ChromaExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            method=chroma_method,
            **kwargs
        )
        
        self.mfcc_extractor = None
        if include_mfcc:
            self.mfcc_extractor = MFCCExtractor(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                **kwargs
            )
        
        self.spectral_extractor = None
        if include_spectral:
            self.spectral_extractor = SpectralExtractor(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                **kwargs
            )
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Dictionary of extracted features.
        """
        features = {}
        
        # Extract chroma features
        features["chroma"] = self.chroma_extractor.extract(audio)
        
        # Extract MFCC features
        if self.mfcc_extractor:
            features["mfcc"] = self.mfcc_extractor.extract(audio)
        
        # Extract spectral features
        if self.spectral_extractor:
            spectral_features = self.spectral_extractor.extract(audio)
            features.update(spectral_features)
        
        return features
    
    def extract_mean_features(self, audio: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Extract mean features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Dictionary of mean extracted features.
        """
        features = {}
        
        # Extract mean chroma features
        features["chroma"] = self.chroma_extractor.extract_mean(audio)
        
        # Extract mean MFCC features
        if self.mfcc_extractor:
            features["mfcc"] = self.mfcc_extractor.extract_mean(audio)
        
        # Extract mean spectral features
        if self.spectral_extractor:
            spectral_features = self.spectral_extractor.extract_mean(audio)
            features.update(spectral_features)
        
        return features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of extracted features.
        
        Returns:
            Dictionary of feature dimensions.
        """
        dimensions = {"chroma": 12}
        
        if self.mfcc_extractor:
            dimensions["mfcc"] = self.mfcc_extractor.n_mfcc
        
        if self.spectral_extractor:
            spectral_count = sum([
                self.spectral_extractor.include_centroid,
                self.spectral_extractor.include_rolloff,
                self.spectral_extractor.include_bandwidth,
                self.spectral_extractor.include_zcr,
            ])
            dimensions["spectral"] = spectral_count
        
        return dimensions
