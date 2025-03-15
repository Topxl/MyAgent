"""
Noise Reduction implementation using noisereduce library.
"""
import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import librosa
import noisereduce as nr


class NoiseReduction:
    """
    Noise Reduction class that provides methods to reduce background noise from audio.
    """
    
    def __init__(self, stationary_threshold: float = 0.5, n_fft: int = 2048):
        """
        Initialize the noise reduction module.
        
        Args:
            stationary_threshold: Threshold for stationary noise detection.
            n_fft: FFT window size for spectral noise reduction.
        """
        self.stationary_threshold = stationary_threshold
        self.n_fft = n_fft
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        return audio_data, sample_rate
    
    def _save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str):
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            output_path: Path to save the processed audio.
        """
        sf.write(output_path, audio_data, sample_rate)
    
    def reduce_noise_spectral(self, audio_data: np.ndarray, sample_rate: int, 
                             noise_clip: Optional[np.ndarray] = None, 
                             prop_decrease: float = 0.8) -> np.ndarray:
        """
        Apply spectral noise reduction using the noisereduce library.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            noise_clip: Optional noise profile to use for noise reduction.
            prop_decrease: Proportion to decrease the noise by (0.0 to 1.0).
            
        Returns:
            Noise-reduced audio data as numpy array.
        """
        # If noise_clip is provided, use it for noise reduction
        if noise_clip is not None:
            reduced_audio = nr.reduce_noise(
                y=audio_data, 
                sr=sample_rate,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                n_fft=self.n_fft
            )
        else:
            # Otherwise, use stationary noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio_data, 
                sr=sample_rate,
                stationary=True,
                prop_decrease=prop_decrease,
                n_fft=self.n_fft
            )
        
        return reduced_audio
    
    def process(self, audio_path: str, output_path: Optional[str] = None, 
               noise_clip_path: Optional[str] = None) -> np.ndarray:
        """
        Process an audio file to reduce noise.
        
        Args:
            audio_path: Path to the audio file to process.
            output_path: Optional path to save the processed audio.
            noise_clip_path: Optional path to an audio file containing noise profile.
            
        Returns:
            Noise-reduced audio data as numpy array.
        """
        # Load the audio file
        audio_data, sample_rate = self._load_audio(audio_path)
        
        # Load noise clip if provided
        noise_clip = None
        if noise_clip_path:
            noise_clip, _ = self._load_audio(noise_clip_path)
        
        # Apply noise reduction
        reduced_audio = self.reduce_noise_spectral(
            audio_data=audio_data,
            sample_rate=sample_rate,
            noise_clip=noise_clip
        )
        
        # Save the processed audio if output path is provided
        if output_path:
            self._save_audio(reduced_audio, sample_rate, output_path)
        
        return reduced_audio
