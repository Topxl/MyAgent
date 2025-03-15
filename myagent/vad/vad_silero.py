"""
Voice Activity Detection using Silero VAD model.
"""
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import urllib.request


class VAD:
    """
    Voice Activity Detection using Silero VAD model.
    Automatically identifies when human speech is present to prevent unnecessary continuous recording.
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5, sampling_rate: int = 16000):
        """
        Initialize the VAD model.
        
        Args:
            model_path: Path to a saved Silero VAD model. If None, the model will be downloaded.
            threshold: Detection threshold between 0 and 1. Higher values make detection more strict.
            sampling_rate: Audio sampling rate (default: 16000 Hz).
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        
        # Load or download the model
        if model_path and os.path.exists(model_path):
            self.model = torch.jit.load(model_path)
        else:
            print("Downloading Silero VAD model...")
            model_url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit"
            
            # Create a directory for models if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Download the model
            self.model_path = os.path.join(models_dir, "silero_vad.jit")
            if not os.path.exists(self.model_path):
                urllib.request.urlretrieve(model_url, self.model_path)
            
            self.model = torch.jit.load(self.model_path)
        
        self.model.eval()
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for the VAD model.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Preprocessed audio tensor.
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def detect(self, audio_path: str, window_size_ms: int = 96) -> List[Dict[str, float]]:
        """
        Detect speech segments in an audio file.
        
        Args:
            audio_path: Path to audio file.
            window_size_ms: Size of the sliding window in milliseconds.
        
        Returns:
            List of dictionaries containing start and end times of detected speech segments.
        """
        waveform = self._preprocess_audio(audio_path)
        
        # Get number of samples per window
        window_size_samples = int(self.sampling_rate * window_size_ms / 1000)
        
        # Process audio in windows and get speech probabilities
        speech_probs = []
        
        for i in range(0, waveform.shape[1], window_size_samples):
            window = waveform[:, i:i+window_size_samples]
            
            # Pad the last window if necessary
            if window.shape[1] < window_size_samples:
                padding = window_size_samples - window.shape[1]
                window = torch.nn.functional.pad(window, (0, padding))
            
            with torch.no_grad():
                speech_prob = self.model(window, self.sampling_rate).item()
            speech_probs.append(speech_prob)
        
        # Detect speech segments
        speech_segments = []
        is_speech = False
        start_time = 0
        
        for i, prob in enumerate(speech_probs):
            current_time = i * window_size_ms / 1000  # Convert to seconds
            
            if prob >= self.threshold and not is_speech:
                # Speech start
                is_speech = True
                start_time = current_time
            elif prob < self.threshold and is_speech:
                # Speech end
                is_speech = False
                speech_segments.append({
                    "start": start_time,
                    "end": current_time
                })
        
        # Add the last segment if audio ends during speech
        if is_speech:
            speech_segments.append({
                "start": start_time,
                "end": len(speech_probs) * window_size_ms / 1000
            })
        
        return speech_segments
    
    def detect_realtime(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect speech in a real-time audio chunk.
        
        Args:
            audio_chunk: Numpy array of audio samples.
        
        Returns:
            True if speech is detected, False otherwise.
        """
        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio_chunk).unsqueeze(0)
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
        
        return speech_prob >= self.threshold
