"""
Audio utility functions for MyAgent.
"""
import os
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional


def convert_audio(
    input_path: str, 
    output_path: str, 
    target_sr: int = 16000, 
    target_channels: int = 1,
    format: Optional[str] = None
) -> str:
    """
    Convert audio file to target sample rate and number of channels.
    
    Args:
        input_path: Path to input audio file.
        output_path: Path to save the converted audio.
        target_sr: Target sample rate.
        target_channels: Target number of channels (1 for mono, 2 for stereo).
        format: Output format (inferred from output_path extension if None).
        
    Returns:
        Path to the converted audio file.
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=(target_channels == 1))
    
    # Convert sample rate if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Handle channels
    if target_channels == 1 and audio.ndim > 1:
        # Convert to mono by averaging channels
        audio = np.mean(audio, axis=0)
    elif target_channels == 2 and audio.ndim == 1:
        # Duplicate mono channel to stereo
        audio = np.vstack((audio, audio))
    
    # Save the converted audio
    sf.write(output_path, audio, target_sr, format=format)
    
    return output_path


def normalize_audio(
    audio_data: np.ndarray, 
    target_peak: float = 0.5,
    target_rms: Optional[float] = None
) -> np.ndarray:
    """
    Normalize audio data to a target peak level or RMS.
    
    Args:
        audio_data: Audio data as numpy array.
        target_peak: Target peak amplitude (0.0 to 1.0).
        target_rms: Target RMS level (if provided, overrides target_peak).
        
    Returns:
        Normalized audio data.
    """
    if target_rms is not None:
        # Normalize by RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms > 0:
            gain = target_rms / rms
            normalized = audio_data * gain
        else:
            normalized = audio_data
    else:
        # Normalize by peak
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            gain = target_peak / peak
            normalized = audio_data * gain
        else:
            normalized = audio_data
    
    return normalized


def split_audio(
    audio_data: np.ndarray, 
    sr: int, 
    segment_length_sec: float = 10.0,
    overlap_sec: float = 1.0
) -> List[np.ndarray]:
    """
    Split audio data into segments of fixed length with overlap.
    
    Args:
        audio_data: Audio data as numpy array.
        sr: Sample rate of the audio.
        segment_length_sec: Length of each segment in seconds.
        overlap_sec: Overlap between segments in seconds.
        
    Returns:
        List of audio segments as numpy arrays.
    """
    # Calculate segment length and overlap in samples
    segment_length = int(segment_length_sec * sr)
    overlap = int(overlap_sec * sr)
    hop_length = segment_length - overlap
    
    # Ensure hop_length is positive
    if hop_length <= 0:
        raise ValueError("Segment length must be greater than overlap")
    
    # Split audio
    segments = []
    for i in range(0, len(audio_data), hop_length):
        end = min(i + segment_length, len(audio_data))
        segment = audio_data[i:end]
        
        # Only add segments that are at least half the desired length
        if len(segment) >= segment_length / 2:
            segments.append(segment)
        
        # Stop if we've reached the end of the audio
        if end == len(audio_data):
            break
    
    return segments


def segment_to_timestamp(segment_idx: int, segment_length_sec: float, overlap_sec: float) -> Tuple[float, float]:
    """
    Convert segment index to timestamp.
    
    Args:
        segment_idx: Index of the segment.
        segment_length_sec: Length of each segment in seconds.
        overlap_sec: Overlap between segments in seconds.
        
    Returns:
        Tuple of (start_time, end_time) in seconds.
    """
    hop_length = segment_length_sec - overlap_sec
    start_time = segment_idx * hop_length
    end_time = start_time + segment_length_sec
    
    return start_time, end_time


def detect_silence(
    audio_data: np.ndarray, 
    sr: int, 
    min_silence_duration_sec: float = 0.5,
    silence_threshold_db: float = -40
) -> List[Dict[str, float]]:
    """
    Detect silent segments in audio.
    
    Args:
        audio_data: Audio data as numpy array.
        sr: Sample rate of the audio.
        min_silence_duration_sec: Minimum duration of silence in seconds.
        silence_threshold_db: Threshold for silence detection in dB.
        
    Returns:
        List of dictionaries containing start and end times of silent segments.
    """
    # Convert threshold from dB to amplitude
    silence_threshold = 10 ** (silence_threshold_db / 20)
    
    # Calculate frame energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Get frame energy
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find silent frames
    silent_frames = energy < silence_threshold
    
    # Group silent frames into segments
    silent_segments = []
    is_silent = False
    start_frame = 0
    
    for i, silent in enumerate(silent_frames):
        if silent and not is_silent:
            # Start of silence
            is_silent = True
            start_frame = i
        elif not silent and is_silent:
            # End of silence
            is_silent = False
            end_frame = i
            
            # Calculate duration
            duration_sec = (end_frame - start_frame) * hop_length / sr
            
            # Only keep silences longer than the minimum duration
            if duration_sec >= min_silence_duration_sec:
                silent_segments.append({
                    "start": start_frame * hop_length / sr,
                    "end": end_frame * hop_length / sr
                })
    
    # Add the last segment if still in silence
    if is_silent:
        end_frame = len(silent_frames)
        duration_sec = (end_frame - start_frame) * hop_length / sr
        
        if duration_sec >= min_silence_duration_sec:
            silent_segments.append({
                "start": start_frame * hop_length / sr,
                "end": end_frame * hop_length / sr
            })
    
    return silent_segments
