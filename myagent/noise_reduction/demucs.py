"""
Advanced noise reduction using Demucs (Deep Extractor for Music Sources).
"""
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Optional, Tuple, List
import warnings

# Suppress warnings that might occur during Demucs import
warnings.filterwarnings("ignore")


class DemucsNoiseReducer:
    """
    Advanced noise reduction using Demucs for voice isolation.
    Demucs is a state-of-the-art music source separation model that can
    separate vocals from background noise.
    """
    
    def __init__(self, model_name: str = "htdemucs", device: str = "cpu"):
        """
        Initialize the Demucs noise reducer.
        
        Args:
            model_name: Name of the Demucs model to use.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the Demucs model.
        """
        try:
            # Delayed import to avoid dependency issues
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Store apply_model function for later use
            self.apply_model = apply_model
            
            print(f"Loaded Demucs model: {self.model_name}")
        except ImportError:
            print("Demucs not installed. Please install it with:")
            print("pip install demucs")
            raise
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sample_rate = librosa.load(audio_path, sr=44100, mono=False)
        
        # If mono, reshape to stereo format expected by Demucs
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])
        
        return audio_data, sample_rate
    
    def _save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str):
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            output_path: Path to save the processed audio.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # If stereo, convert to mono by averaging channels
        if audio_data.ndim > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        sf.write(output_path, audio_data, sample_rate)
    
    def separate_sources(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """
        Separate the audio into different sources using Demucs.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Dictionary of separated sources.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please ensure Demucs is installed.")
        
        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        
        # Apply the model
        with torch.no_grad():
            sources = self.apply_model(self.model, audio_tensor, device=self.device)
        
        # Convert back to numpy
        sources_np = {}
        for source, source_tensor in sources.items():
            sources_np[source] = source_tensor.cpu().numpy()
        
        return sources_np
    
    def process(self, audio_path: str, output_path: Optional[str] = None, 
                only_vocals: bool = True) -> np.ndarray:
        """
        Process an audio file to extract vocals and reduce noise.
        
        Args:
            audio_path: Path to the audio file to process.
            output_path: Optional path to save the processed audio.
            only_vocals: If True, return only the vocals. If False, return all sources.
            
        Returns:
            Processed audio data as numpy array (vocals only by default).
        """
        # Load the audio file
        audio_data, sample_rate = self._load_audio(audio_path)
        
        # Separate sources
        sources = self.separate_sources(audio_data, sample_rate)
        
        # Save the processed audio if output path is provided
        if output_path and only_vocals:
            self._save_audio(sources['vocals'], sample_rate, output_path)
            result = sources['vocals']
        elif output_path and not only_vocals:
            # Create a directory for all sources
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            for source_name, source_data in sources.items():
                source_path = os.path.join(base_dir, f"{base_name}_{source_name}.wav")
                self._save_audio(source_data, sample_rate, source_path)
            
            # Return vocals by default
            result = sources['vocals']
        else:
            # Just return the vocals if no output path is provided
            result = sources['vocals'] if only_vocals else sources
        
        return result
